# TrimeshLoader

An executor that wraps trimesh for loading triangular meshes, and extract cloud points.

Documents can either have
- a `uri`: that represents
    - the path (e.g., `local disk`, `http(s)` path) to the mesh data file or
    - the `base64` content, and the file format is specified in document tags, i.e., `doc.tags["file_format"]`.
- a `blob` obtained after converting the uri to buffer, and the format is specified in document tags, e.g., `doc.tags["file_format"]="glb"`.

`TrimeshLoader` samples points from the 3D mesh object to create a point cloud and puts them in the tensor attribute of the Document.


## Usage

#### via Docker image (recommended)

```python
from jina import Flow

f = Flow().add(uses='jinahub+docker://TrimeshLoader')
```

#### via source code

```python
from jina import Flow

f = Flow().add(uses='jinahub://TrimeshLoader')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
