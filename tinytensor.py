import numpy as np
import string


# Operations that broadcast their inputs (e.g. (n, m) + (m,) -> (n, m))
# produce an output gradient with extra or stretched axes. Sum it back
# down to the shape of the input before accumulating.
def _unbroadcast(grad: np.ndarray, shape: tuple):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data: np.ndarray | list, calculateGrad: bool = False):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=np.double)

        self.shape = np.shape(self.data)
        self.ndim = np.ndim(self.data)
        self.grad = np.zeros(self.shape)
        self.children: list[Tensor] = []
        self.calculateGrad = calculateGrad
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data=\n{self.data})"

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t: Tensor):
            if t not in visited:
                visited.add(t)
                for child in t.children:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

    # --- Arithmetic operations ---
    # Element-wise addition
    def __add__(self, other):
        out = Tensor(self.data + other.data)
        out.children += [self, other]
        out.calculateGrad = self.calculateGrad or other.calculateGrad

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape) if self.calculateGrad else 0
            other.grad += _unbroadcast(out.grad, other.shape) if other.calculateGrad else 0

        out._backward = _backward
        return out

    # Element-wise multiplication
    def __mul__(self, other):
        out = Tensor(self.data * other.data)
        out.children += [self, other]
        out.calculateGrad = self.calculateGrad or other.calculateGrad

        def _backward():
            self.grad += (
                _unbroadcast(other.data * out.grad, self.shape) if self.calculateGrad else 0
            )
            other.grad += (
                _unbroadcast(self.data * out.grad, other.shape) if other.calculateGrad else 0
            )

        out._backward = _backward
        return out

    def __neg__(self):
        return self * Tensor([-1])

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # Matrix and vector multiplication (operands with at most 2 axes)
    def __matmul__(self, other):
        left = "ij" if self.ndim == 2 else "j"
        right = "jk" if other.ndim == 2 else "j"
        out = ("i" if self.ndim == 2 else "") + ("k" if other.ndim == 2 else "")
        schema = f"{left},{right}->{out}"

        out = contraction([self, other], schema)
        out.children += [self, other]
        out.calculateGrad = self.calculateGrad or other.calculateGrad

        def _backward():
            self.grad += (
                _partial_gradient([self.data, other.data], schema, 0, out.grad)
                if self.calculateGrad
                else 0
            )
            other.grad += (
                _partial_gradient([self.data, other.data], schema, 1, out.grad)
                if other.calculateGrad
                else 0
            )

        out._backward = _backward
        return out


def einsum(tensors: list[Tensor], schema: str):
    out = contraction(tensors, schema)
    out.children += tensors
    out.calculateGrad = any(t.calculateGrad for t in tensors)

    def _backward():
        for pos, t in enumerate(tensors):
            t.grad += (
                _partial_gradient([s.data for s in tensors], schema, pos, out.grad)
                if t.calculateGrad
                else 0
            )

    out._backward = _backward
    return out


def relu(tensor: Tensor):
    out = Tensor(np.maximum(0, tensor.data))
    out.children.append(tensor)
    out.calculateGrad = tensor.calculateGrad

    def _backward():
        tensor.grad += (
            (tensor.data > 0).astype(tensor.data.dtype) * out.grad if tensor.calculateGrad else 0
        )

    out._backward = _backward
    return out


def tanh(tensor: Tensor):
    out = Tensor(np.tanh(tensor.data))
    out.children.append(tensor)
    out.calculateGrad = tensor.calculateGrad

    def _backward():
        tensor.grad += (
            (np.ones_like(out.data) - (out.data * out.data)) * out.grad
            if tensor.calculateGrad
            else 0
        )

    out._backward = _backward
    return out


def insert(tensor: Tensor, insertion: tuple[int, ...]):
    return Tensor(np.expand_dims(tensor.data, insertion))


def stretch(tensor: Tensor, insertion: tuple[int, ...], length: tuple[int, ...]) -> Tensor:
    array = np.expand_dims(tensor.data, insertion)

    shape = list(tensor.shape)
    for i, val in zip(insertion, length):
        shape.insert(i, val)
    out = Tensor(np.broadcast_to(array, shape))
    return out


def contraction(tensors: list[Tensor], schema: str):
    return Tensor(np.einsum(schema, *[t.data for t in tensors]))


def partial_jacobian(tensors: list[Tensor], schema: str, targetPos: int):
    # Find letters that are not used in the schema
    lhs, rhs = schema.replace(" ", "").split("->")
    inds = lhs.split(",")
    l_used, r_used = set(lhs.replace(",", "")), set(rhs)
    unused = list(c for c in string.ascii_letters if c not in l_used | r_used)
    unused.sort()

    # Remove the target tensor and its indices from the operands
    _tensors = list(tensors)
    target = _tensors.pop(targetPos)
    targetInd = inds.pop(targetPos)

    # Construct the input and output indices
    outInd = rhs
    trailing_dims = {}
    for pos, (i, p) in enumerate(zip(targetInd, unused)):
        # Append a fresh output index for this axis of the target
        outInd += p
        trailing_dims[p] = target.shape[pos]

        # Adjust the input indices
        # Case: the target index also appears in the output
        if i in rhs:
            axis_length = target.shape[pos]
            _tensors.append(Tensor(np.identity(axis_length)))
            inds.append(i + p)

        # Case: the target index does not appear in the output
        else:
            inds = [ind.replace(i, p) for ind in inds]

    # Find the output indices that occur in none of the inputs, and their positions
    inds_set = set("".join(inds))
    missing = set(outInd) - inds_set

    insertions = tuple(pos for pos, i in enumerate(outInd) if i in missing)
    occuring_out = "".join([i for i in outInd if i in inds_set])

    # Contract over the indices that do occur in the inputs
    if inds:  # np.einsum needs at least one operand
        occuring_result = np.einsum(
            f"{",".join(inds)} -> {occuring_out}", *[t.data for t in _tensors]
        )
    else:
        occuring_result = np.array(1.0)

    # Insert the missing axes and broadcast them to the right length
    final_shape = (
        trailing_dims[p] if p in trailing_dims else np.shape(occuring_result)[pos]
        for pos, p in enumerate(outInd)
    )

    inserted_result = np.expand_dims(occuring_result, insertions)
    final_result = np.broadcast_to(inserted_result, final_shape)

    return Tensor(final_result)


def partial_gradient(tensors: list[Tensor], schema: str, targetPos: int, outGrad: Tensor):
    lhs, rhs = schema.replace(" ", "").split("->")
    inds = lhs.split(",")

    # Remove the target tensor and its indices from the operands
    _tensors = list(tensors)
    target = _tensors.pop(targetPos)
    targetInd = inds.pop(targetPos)

    # The output gradient becomes an operand, and the result takes the target's indices
    outInd = targetInd
    inds.append(rhs)
    _tensors.append(outGrad)

    # Find the output indices that occur in none of the inputs, and their positions
    inds_set = set("".join(inds))
    missing = set(outInd) - inds_set

    insertions = tuple(pos for pos, i in enumerate(outInd) if i in missing)
    occuring_out = "".join([i for i in outInd if i in inds_set])

    # Contract over the indices that do occur in the inputs
    if inds:  # np.einsum needs at least one operand
        occuring_result = np.einsum(
            f"{",".join(inds)} -> {occuring_out}", *[t.data for t in _tensors]
        )
    else:
        occuring_result = np.array(1.0)

    # Insert the missing axes and broadcast them to the right length
    final_shape = target.shape

    inserted_result = np.expand_dims(occuring_result, insertions)
    final_result = np.broadcast_to(inserted_result, final_shape)

    return Tensor(final_result)


def _partial_gradient(tensors: list[np.ndarray], schema: str, targetPos: int, outGrad: np.ndarray):
    lhs, rhs = schema.replace(" ", "").split("->")
    inds = lhs.split(",")

    # Remove the target tensor and its indices from the operands
    _tensors = list(tensors)
    target = _tensors.pop(targetPos)
    targetInd = inds.pop(targetPos)

    # The output gradient becomes an operand, and the result takes the target's indices
    outInd = targetInd
    inds.append(rhs)
    _tensors.append(outGrad)

    # Find the output indices that occur in none of the inputs, and their positions
    inds_set = set("".join(inds))
    missing = set(outInd) - inds_set

    insertions = tuple(pos for pos, i in enumerate(outInd) if i in missing)
    occuring_out = "".join([i for i in outInd if i in inds_set])

    # Contract over the indices that do occur in the inputs
    if inds:  # np.einsum needs at least one operand
        occuring_result = np.einsum(f"{",".join(inds)} -> {occuring_out}", *[t for t in _tensors])
    else:
        occuring_result = np.array(1.0)

    # Insert the missing axes and broadcast them to the right length
    final_shape = np.shape(target)

    inserted_result = np.expand_dims(occuring_result, insertions)
    final_result = np.broadcast_to(inserted_result, final_shape)

    return final_result
