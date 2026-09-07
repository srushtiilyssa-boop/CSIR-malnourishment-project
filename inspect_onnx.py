import onnx


MODEL_PATH = "assets/models/malnutrition_model.onnx"


print("========================================")
print("ONNX MODEL INSPECTION")
print("========================================")

model = onnx.load(MODEL_PATH)

print("\nGRAPH INPUTS")
print("----------------------------------------")

for inp in model.graph.input:
    print("Name:", inp.name)
    print("Type:", inp.type)


print("\nGRAPH OUTPUTS")
print("----------------------------------------")

for out in model.graph.output:
    print("Name:", out.name)
    print("Type:", out.type)

    if out.type.HasField("tensor_type"):
        tensor_type = out.type.tensor_type

        print(
            "  Tensor element type:",
            tensor_type.elem_type,
        )

        if tensor_type.HasField("shape"):
            print(
                "  Shape:",
                tensor_type.shape,
            )

    if out.type.HasField("sequence_type"):
        print("  >>> SEQUENCE OUTPUT <<<")

    if out.type.HasField("map_type"):
        print("  >>> MAP OUTPUT <<<")


print("\nNODES")
print("----------------------------------------")

for node in model.graph.node:
    print(
        f"{node.op_type}: "
        f"{list(node.input)} -> {list(node.output)}"
    )


print("\nMODEL CHECK")
print("----------------------------------------")

onnx.checker.check_model(model)

print("Model is valid.")

print("\n========================================")
print("DONE")
print("========================================")