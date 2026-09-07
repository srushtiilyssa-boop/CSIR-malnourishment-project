import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

class PredictionResult {
  final String label;
  final double confidence;
  final int classIndex;
  final List<double> probabilities;

  PredictionResult({
    required this.label,
    required this.confidence,
    required this.classIndex,
    required this.probabilities,
  });
}

class OnnxService {
  late final OnnxRuntime _onnx;

  OrtSession? scalerSession;
  OrtSession? modelSession;

  Future<void> initialize() async {
    _onnx = OnnxRuntime();

    final options = OrtSessionOptions();

    // ==========================================================
    // LOAD SCALER
    // ==========================================================

    scalerSession = await _onnx.createSessionFromAsset(
      'assets/models/scaler.onnx',
      options: options,
    );

    // ==========================================================
    // LOAD FIXED MALNUTRITION MODEL
    // ==========================================================

    modelSession = await _onnx.createSessionFromAsset(
      'assets/models/malnutrition_model_fixed.onnx',
      options: options,
    );

    print("=================================");
    print("ONNX INITIALIZATION");
    print("=================================");

    print("MODEL INPUT INFO:");
    print(await modelSession!.getInputInfo());

    print("---------------------------------");

    print("MODEL OUTPUT INFO:");
    print(await modelSession!.getOutputInfo());

    print("---------------------------------");

    print("SCALER INPUT INFO:");
    print(await scalerSession!.getInputInfo());

    print("---------------------------------");

    print("SCALER OUTPUT INFO:");
    print(await scalerSession!.getOutputInfo());

    print("=================================");
    print("ONNX MODELS LOADED");
    print("=================================");
  }

  Future<PredictionResult> predict(
    List<double> features,
  ) async {
    if (scalerSession == null ||
        modelSession == null) {
      throw Exception(
        "ONNX models are not initialized.",
      );
    }

    // ==========================================================
    // CHECK FEATURE COUNT
    // ==========================================================

    if (features.length != 23) {
      throw Exception(
        "Expected exactly 23 features, "
        "but received ${features.length}.",
      );
    }

    // ==========================================================
    // CHECK FOR INVALID VALUES
    // ==========================================================

    for (int i = 0; i < features.length; i++) {
      if (features[i].isNaN ||
          features[i].isInfinite) {
        throw Exception(
          "Invalid feature at index $i: ${features[i]}",
        );
      }
    }

    // ==========================================================
    // PRINT RAW FEATURES
    // ==========================================================

    print("");
    print("=================================");
    print("RAW PATIENT FEATURES");
    print("=================================");

    for (int i = 0; i < features.length; i++) {
      print(
        "Feature ${i + 1}: ${features[i]}",
      );
    }

    print(
      "FEATURE COUNT: ${features.length}",
    );

    // ==========================================================
    // CREATE INPUT TENSOR
    // ==========================================================

    final inputTensor = await OrtValue.fromList(
      features,
      [1, 23],
    );

    Map<String, OrtValue>? scalerOutputs;
    Map<String, OrtValue>? modelOutputs;

    try {
      // ========================================================
      // STEP 1 — SCALER
      // ========================================================

      print("");
      print("=================================");
      print("RUNNING SCALER");
      print("=================================");

      final scalerInputName =
          scalerSession!.inputNames.first;

      scalerOutputs = await scalerSession!.run({
        scalerInputName: inputTensor,
      });

      if (scalerOutputs.isEmpty) {
        throw Exception(
          "Scaler returned no output.",
        );
      }

      final scaledTensor =
          scalerOutputs.values.first;

      final scaledData =
          await scaledTensor.asFlattenedList();

      print("SCALED FEATURES:");
      print(scaledData);

      // ========================================================
      // STEP 2 — MALNUTRITION MODEL
      // ========================================================

      print("");
      print("=================================");
      print("RUNNING MALNUTRITION MODEL");
      print("=================================");

      final modelInputName =
          modelSession!.inputNames.first;

      modelOutputs = await modelSession!.run({
        modelInputName: scaledTensor,
      });

      if (modelOutputs.isEmpty) {
        throw Exception(
          "Malnutrition model returned no outputs.",
        );
      }

      // ========================================================
      // PRINT OUTPUTS
      // ========================================================

      print("");
      print("=================================");
      print("MODEL OUTPUTS");
      print("=================================");

      print(
        "Output names: ${modelOutputs.keys}",
      );

      print("=================================");

      // ========================================================
      // LABEL
      // ========================================================

      final labelOutput =
          modelOutputs['label'];

      if (labelOutput == null) {
        throw Exception(
          "Model did not return 'label'. "
          "Available outputs: "
          "${modelOutputs.keys}",
        );
      }

      final labelData =
          await labelOutput.asFlattenedList();

      if (labelData.isEmpty) {
        throw Exception(
          "Model returned an empty label.",
        );
      }

      final rawLabel = labelData.first;

      final classIndex =
          (rawLabel as num).toInt();

      print("");
      print("=================================");
      print("RAW MODEL LABEL");
      print("=================================");

      print(labelData);

      print(
        "Class index: $classIndex",
      );

      // ========================================================
      // PROBABILITIES
      // ========================================================

      final probabilityOutput =
          modelOutputs['probabilities'];

      if (probabilityOutput == null) {
        throw Exception(
          "Model did not return 'probabilities'. "
          "Available outputs: "
          "${modelOutputs.keys}",
        );
      }

      final probabilityData =
          await probabilityOutput.asFlattenedList();

      final probabilities =
          probabilityData
              .whereType<num>()
              .map(
                (value) => value.toDouble(),
              )
              .toList();

      if (probabilities.isEmpty) {
        throw Exception(
          "Model returned empty probabilities.",
        );
      }

      print("");
      print("=================================");
      print("RAW MODEL PROBABILITIES");
      print("=================================");

      print(probabilities);

      // ========================================================
      // CONFIDENCE
      // ========================================================

      double confidence = 0.0;

      if (classIndex >= 0 &&
          classIndex < probabilities.length) {
        confidence =
            probabilities[classIndex];
      }

      // Make sure confidence stays valid.
      confidence =
          confidence.clamp(0.0, 1.0);

      // ========================================================
      // TEMPORARY LABEL
      // ========================================================
      //
      // We are intentionally NOT calling Class 0
      // "malnourished" or "nourished" yet.
      //
      // We first need to verify the actual class mapping
      // from the training/model information.
      // ========================================================

      String prediction;

      if (classIndex == 0) {
        prediction = "Nourished";
      } else if (classIndex == 1) {
        prediction = "Malnourished";
      } else {
        prediction = "Unknown";
      }
     
      // ========================================================
      // FINAL RESULT
      // ========================================================

      print("");
      print("=================================");
      print("FINAL MODEL RESULT");
      print("=================================");

      print(
        "MODEL CLASS: $classIndex",
      );

      print(
        "USER-FRIENDLY PREDICTION: $prediction",
      );

      print(
        "CONFIDENCE: "
        "${(confidence * 100).toStringAsFixed(2)}%",
      );

      print(
        "PROBABILITIES: $probabilities",
      );

      print("=================================");
      print("");

      return PredictionResult(
        label: prediction,
        confidence: confidence,
        classIndex: classIndex,
        probabilities: probabilities,
      );
    } finally {
      // ========================================================
      // CLEANUP
      // ========================================================

      await inputTensor.dispose();

      if (scalerOutputs != null) {
        for (final output
            in scalerOutputs.values) {
          await output.dispose();
        }
      }

      if (modelOutputs != null) {
        for (final output
            in modelOutputs.values) {
          await output.dispose();
        }
      }
    }
  }
}