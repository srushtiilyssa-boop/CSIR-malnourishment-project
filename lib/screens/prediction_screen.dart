import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../services/onnx_service.dart';
import '../services/history_service.dart';
import 'result_screen.dart';

class PredictionScreen extends StatefulWidget {
  final Patient patient;

  const PredictionScreen({
    super.key,
    required this.patient,
  });

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final OnnxService _onnx = OnnxService();

  String status = "Analyzing Child Data...";

  @override
  void initState() {
    super.initState();
    _predict();
  }

  Future<void> _predict() async {
    try {
      // ============================================================
      // 1. LOAD ONNX MODEL
      // ============================================================

      if (mounted) {
        setState(() {
          status = "Loading prediction model...";
        });
      }

      await _onnx.initialize();

      // ============================================================
      // 2. CREATE EXACT 23-FEATURE VECTOR
      // ============================================================
      //
      // IMPORTANT:
      // Patient.toFeatureVector() is the SINGLE source of truth.
      //
      // Feature order:
      //
      // 1  child_age_months
      // 2  gender
      // 3  birth_weight
      // 4  weight_kg
      // 5  height_cm
      // 6  breastfeeding
      // 7  diarrhea
      // 8  fever
      // 9  cough
      // 10 mother_age
      // 11 mother_education
      // 12 mother_bmi
      // 13 children_ever_born
      // 14 birth_order
      // 15 urban_rural
      // 16 wealth_index
      // 17 water_source
      // 18 time_to_water
      // 19 sanitation_toilet_facility
      // 20 handwashing_facility
      // 21 child_age_years
      // 22 household_members
      // 23 antenatal_visits
      //
      // This matches the feature order used by the Python model.
      // ============================================================

      final features = widget.patient.toFeatureVector();

      // ============================================================
      // 3. SAFETY CHECK
      // ============================================================

      if (features.length != 23) {
        throw Exception(
          "INVALID FEATURE COUNT: Expected 23 features, "
          "but received ${features.length}.",
        );
      }

      // ============================================================
      // 4. PRINT EXACT FEATURES SENT TO MODEL
      // ============================================================

      const featureNames = <String>[
        "child_age_months",
        "gender",
        "birth_weight",
        "weight_kg",
        "height_cm",
        "breastfeeding",
        "diarrhea",
        "fever",
        "cough",
        "mother_age",
        "mother_education",
        "mother_bmi",
        "children_ever_born",
        "birth_order",
        "urban_rural",
        "wealth_index",
        "water_source",
        "time_to_water",
        "sanitation_toilet_facility",
        "handwashing_facility",
        "child_age_years",
        "household_members",
        "antenatal_visits",
      ];

      print("");
      print("======================================================");
      print("EXACT PATIENT DATA SENT TO ML MODEL");
      print("======================================================");

      for (int i = 0; i < features.length; i++) {
        print(
          "${i + 1}. ${featureNames[i]} = ${features[i]}",
        );
      }

      print("------------------------------------------------------");
      print("TOTAL FEATURES: ${features.length}");
      print("------------------------------------------------------");
      print("FEATURE VECTOR:");
      print(features);
      print("======================================================");
      print("");

      // ============================================================
      // 5. UPDATE SCREEN
      // ============================================================

      if (mounted) {
        setState(() {
          status = "Running ML prediction...";
        });
      }

      // ============================================================
      // 6. REAL ONNX INFERENCE
      // ============================================================

      final result = await _onnx.predict(features);

      // Save this completed screening to local history.
      await HistoryService.saveScreening(
        patient: widget.patient,
        prediction: result.label,
        confidence: result.confidence,
        classIndex: result.classIndex,
        probabilities: result.probabilities,
      );

      if (!mounted) {
        return;
      }

      // ============================================================
      // 7. PRINT RESULT
      // ============================================================

      print("");
      print("======================================================");
      print("FINAL PREDICTION");
      print("======================================================");
      print("Class Index : ${result.classIndex}");
      print("Label       : ${result.label}");
      print(
        "Confidence  : "
        "${(result.confidence * 100).toStringAsFixed(2)}%",
      );
      print("Probabilities: ${result.probabilities}");
      print("======================================================");
      print("");

      // ============================================================
      // 8. OPEN RESULT SCREEN
      // ============================================================

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            prediction: result.label,
            confidence: result.confidence,
            classIndex: result.classIndex,
            probabilities: result.probabilities,
          ),
        ),
      );
    } catch (e, stackTrace) {
      print("");
      print("======================================================");
      print("PREDICTION ERROR");
      print("======================================================");
      print(e);
      print("");
      print("STACK TRACE:");
      print(stackTrace);
      print("======================================================");
      print("");

      if (!mounted) {
        return;
      }

      setState(() {
        status = "Prediction failed";
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            "Prediction failed: $e",
          ),
          duration: const Duration(seconds: 6),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const CircularProgressIndicator(),

              const SizedBox(height: 25),

              Text(
                status,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),

              const SizedBox(height: 12),

              const Text(
                "Please wait while the AI model analyzes the information...",
                style: TextStyle(
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}