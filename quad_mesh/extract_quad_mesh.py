import os
import subprocess
import sys

import igl
import numpy as np
import pyqex
import torch
import trimesh
from scipy import sparse
from scipy.sparse.linalg import lsqr


def _safe_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def _load_triangle_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'Unsupported mesh type loaded from {mesh_path!r}: {type(mesh)!r}')
    if mesh.faces.shape[1] != 3:
        raise ValueError('Quad extraction currently requires a triangle mesh input.')
    return mesh


def _solve_uvs(vertices, triangles, pd1, pd2):
    pd1_combed, pd2_combed = igl.comb_cross_field(vertices, triangles, pd1, pd2)
    mismatch = igl.cross_field_mismatch(vertices, triangles, pd1_combed, pd2_combed, True)
    cuts = igl.cut_mesh_from_singularities(vertices, triangles, mismatch)
    cut_vertices, cut_triangles, _ = igl.cut_mesh(vertices, triangles, cuts)

    num_vertices = cut_vertices.shape[0]
    rows = []
    cols = []
    vals = []
    bu = []
    bv = []
    row = 0

    for face_index, tri in enumerate(cut_triangles):
        face_points = cut_vertices[tri]
        face_u = pd1_combed[face_index]
        face_v = pd2_combed[face_index]

        for local_i, local_j in ((0, 1), (1, 2), (2, 0)):
            vi = int(tri[local_i])
            vj = int(tri[local_j])
            delta = face_points[local_j] - face_points[local_i]
            rows.extend((row, row))
            cols.extend((vi, vj))
            vals.extend((-1.0, 1.0))
            bu.append(float(delta.dot(face_u)))
            bv.append(float(delta.dot(face_v)))
            row += 1

    anchor0 = int(cut_triangles[0, 0])
    anchor1 = int(cut_triangles[0, 1])
    anchor_delta = cut_vertices[anchor1] - cut_vertices[anchor0]

    rows.append(row)
    cols.append(anchor0)
    vals.append(1.0)
    bu.append(0.0)
    bv.append(0.0)
    row += 1

    rows.append(row)
    cols.append(anchor1)
    vals.append(1.0)
    bu.append(float(anchor_delta.dot(pd1_combed[0])))
    bv.append(float(anchor_delta.dot(pd2_combed[0])))
    row += 1

    system = sparse.coo_matrix((vals, (rows, cols)), shape=(row, num_vertices)).tocsr()
    u = lsqr(system, np.asarray(bu, dtype=np.float64))[0]
    v = lsqr(system, np.asarray(bv, dtype=np.float64))[0]

    uv_per_vertex = np.column_stack((u, v))
    uv_per_triangle = uv_per_vertex[cut_triangles]

    edge_lengths = np.concatenate((
        np.linalg.norm(uv_per_triangle[:, 1] - uv_per_triangle[:, 0], axis=1),
        np.linalg.norm(uv_per_triangle[:, 2] - uv_per_triangle[:, 1], axis=1),
        np.linalg.norm(uv_per_triangle[:, 0] - uv_per_triangle[:, 2], axis=1),
    ))
    valid = edge_lengths > 1e-8
    if np.any(valid):
        uv_per_triangle = uv_per_triangle / np.median(edge_lengths[valid])

    return cut_vertices, cut_triangles, uv_per_triangle


def _write_obj(path, vertices, quad_faces):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# NeurCross quad mesh extraction\n')
        for vertex in vertices:
            handle.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for face in quad_faces:
            handle.write(
                f'f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1} {int(face[3]) + 1}\n'
            )


def _mesh_topology_stats(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        'components': len(mesh.split(only_watertight=False)),
        'boundary_edges': int((counts == 1).sum()),
        'nonmanifold_edges': int((counts > 2).sum()),
        'is_watertight': bool(mesh.is_watertight),
    }


def _quadriflow_fallback(mesh_path, output_path, target_faces):
    payload_path = output_path + '.quadriflow.npz'
    script = r"""
import sys
import bpy
import numpy as np

mesh_path, payload_path, target_faces = sys.argv[1], sys.argv[2], int(sys.argv[3])
bpy.ops.wm.read_factory_settings(use_empty=True)
ext = mesh_path.rsplit('.', 1)[-1].lower()
if ext == 'ply':
    bpy.ops.wm.ply_import(filepath=mesh_path)
elif ext == 'obj':
    bpy.ops.wm.obj_import(filepath=mesh_path)
else:
    raise ValueError(f'Unsupported quadriflow fallback input: {mesh_path}')
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.quadriflow_remesh(
    target_faces=max(100, target_faces),
    use_preserve_boundary=True,
    use_preserve_sharp=False,
)
mesh = bpy.context.view_layer.objects.active.data
vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
faces = np.array([list(poly.vertices) for poly in mesh.polygons], dtype=np.int64)
np.savez(payload_path, vertices=vertices, faces=faces)
"""
    command = [
        sys.executable,
        '-c',
        script,
        mesh_path,
        payload_path,
        str(int(max(100, target_faces))),
    ]
    subprocess.run(command, check=True)
    with np.load(payload_path) as payload:
        vertices = np.asarray(payload['vertices'], dtype=np.float64)
        faces = np.asarray(payload['faces'], dtype=np.int64)
    os.remove(payload_path)
    return vertices, faces


def extract_quad_mesh_from_field(mesh_path, alpha, beta, output_path):
    mesh = _load_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)

    if alpha.shape != beta.shape or alpha.shape[1] != 3:
        raise ValueError('alpha and beta must both have shape (num_faces, 3).')
    if alpha.shape[0] != triangles.shape[0]:
        raise ValueError('Cross field face count must match the mesh face count.')

    alpha = _safe_normalize(np.asarray(alpha, dtype=np.float64))
    beta = _safe_normalize(np.asarray(beta, dtype=np.float64))

    cut_vertices, cut_triangles, uvs_per_triangle = _solve_uvs(vertices, triangles, alpha, beta)
    quad_vertices, quad_faces = pyqex.extract_quads(
        np.asarray(cut_vertices, dtype=np.float64),
        np.asarray(cut_triangles, dtype=np.uint32),
        np.asarray(uvs_per_triangle, dtype=np.float64),
    )

    stats = _mesh_topology_stats(quad_vertices, quad_faces)
    extractor_name = 'pyqex'
    if stats['boundary_edges'] > 0 or stats['components'] > 1 or stats['nonmanifold_edges'] > 0:
        quad_vertices, quad_faces = _quadriflow_fallback(
            mesh_path=mesh_path,
            output_path=output_path,
            target_faces=max(1, triangles.shape[0] // 10),
        )
        stats = _mesh_topology_stats(quad_vertices, quad_faces)
        extractor_name = 'quadriflow_fallback'

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    _write_obj(output_path, quad_vertices, quad_faces)

    return {
        'quad_vertices': quad_vertices,
        'quad_faces': quad_faces,
        'cut_vertices': cut_vertices,
        'cut_triangles': cut_triangles,
        'uvs_per_triangle': uvs_per_triangle,
        'output_path': output_path,
        'topology': stats,
        'extractor': extractor_name,
    }


def extract_quad_mesh_from_network(net, recon_dataset, mesh_path, device, output_path):
    net.eval()
    with torch.no_grad():
        points = torch.from_numpy(recon_dataset.points).unsqueeze(0).to(device=device, dtype=torch.float32)
        normals = torch.from_numpy(recon_dataset.mnfld_n).unsqueeze(0).to(device=device, dtype=torch.float32)
        vector_u = torch.from_numpy(recon_dataset.vector_u).unsqueeze(0).to(device=device, dtype=torch.float32)
        vector_v = torch.from_numpy(recon_dataset.vector_v).unsqueeze(0).to(device=device, dtype=torch.float32)

        features = torch.cat((points, normals, vector_u, vector_v), dim=-1)
        _, theta = net(points, points, angle_features=features)
        theta = theta.squeeze(0)
        normals = normals.squeeze(0)
        vector_u = vector_u.squeeze(0)
        vector_v = vector_v.squeeze(0)

        alpha = vector_u * torch.cos(theta) + vector_v * torch.sin(theta)
        alpha = alpha / (alpha.norm(dim=-1, keepdim=True) + 1e-12)
        beta = -vector_u * torch.sin(theta) + vector_v * torch.cos(theta)
        beta = beta / (beta.norm(dim=-1, keepdim=True) + 1e-12)

    return extract_quad_mesh_from_field(
        mesh_path=mesh_path,
        alpha=alpha.detach().cpu().numpy(),
        beta=beta.detach().cpu().numpy(),
        output_path=output_path,
    )
