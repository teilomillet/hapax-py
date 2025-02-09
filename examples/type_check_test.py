from hapax import ops, Graph

# This should fail at import time - missing type hints
@ops(name="bad_op1")
def no_type_hints(x):
    return x + 1

# This should fail at import time - incompatible types
@ops(name="bad_op2")
def wrong_types(x: str) -> int:
    return len(x)

@ops(name="str_op")
def str_op(x: str) -> str:
    return x.upper()

@ops(name="int_op")
def int_op(x: int) -> int:
    return x + 1

# This should fail at graph definition time - type mismatch between operations
graph = (
    Graph("type_mismatch")
    .then(str_op)      # Takes str, returns str
    .then(int_op)      # Takes int, returns int - incompatible!
)

print("If you see this, type checking failed!") 