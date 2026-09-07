import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  final String prediction;
  final double confidence;
  final int classIndex;
  final List<double> probabilities;

  const ResultScreen({
    super.key,
    required this.prediction,
    required this.confidence,
    required this.classIndex,
    required this.probabilities,
  });

  // Class mapping confirmed from the model:
  // 0 = Nourished
  // 1 = Malnourished
  bool get isMalnourished => classIndex == 1;

  Color get resultColor {
    if (classIndex == 0) {
      return Colors.green;
    }

    if (classIndex == 1) {
      return Colors.red;
    }

    return Colors.grey;
  }

  IconData get resultIcon {
    if (classIndex == 0) {
      return Icons.check_circle;
    }

    if (classIndex == 1) {
      return Icons.warning_rounded;
    }

    return Icons.help_outline;
  }

  String className(int index) {
    if (index == 0) {
      return "Nourished";
    }

    if (index == 1) {
      return "Malnourished";
    }

    return "Unknown";
  }

  // --------------------------------------------------
  // RISK LEVEL
  // --------------------------------------------------
  //
  // The risk is based on the model's predicted class
  // and its confidence.
  //
  // Malnourished:
  //   >= 80% confidence -> HIGH
  //   50-79.99%         -> MEDIUM
  //   < 50%             -> LOW
  //
  // Nourished:
  //   >= 80% confidence -> LOW
  //   50-79.99%         -> MEDIUM
  //   < 50%             -> HIGH
  //
  String get riskLevel {
    final percent = confidence * 100;

    if (isMalnourished) {
      if (percent >= 80) {
        return "HIGH";
      }

      if (percent >= 50) {
        return "MEDIUM";
      }

      return "LOW";
    }

    if (percent >= 80) {
      return "LOW";
    }

    if (percent >= 50) {
      return "MEDIUM";
    }

    return "HIGH";
  }

  Color get riskColor {
    if (riskLevel == "HIGH") {
      return Colors.red;
    }

    if (riskLevel == "MEDIUM") {
      return Colors.orange;
    }

    return Colors.green;
  }

  IconData get riskIcon {
    if (riskLevel == "HIGH") {
      return Icons.warning_rounded;
    }

    if (riskLevel == "MEDIUM") {
      return Icons.info_outline;
    }

    return Icons.check_circle_outline;
  }

  String get riskMessage {
    if (riskLevel == "HIGH") {
      return isMalnourished
          ? "The child may require prompt "
            "nutritional assessment."
          : "The prediction has low confidence. "
            "Consider further assessment.";
    }

    if (riskLevel == "MEDIUM") {
      return "The prediction has moderate confidence. "
          "Consider additional assessment.";
    }

    return "The child is predicted to be nourished "
        "with high model confidence.";
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Nutrition Assessment",
        ),
        centerTitle: true,
      ),

      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),

          children: [
            const SizedBox(height: 10),

            // --------------------------------------------------
            // RESULT ICON
            // --------------------------------------------------
            Center(
              child: Icon(
                resultIcon,
                color: resultColor,
                size: 90,
              ),
            ),

            const SizedBox(height: 15),

            // --------------------------------------------------
            // PREDICTION
            // --------------------------------------------------
            Center(
              child: Text(
                prediction,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: resultColor,
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            const SizedBox(height: 10),

            // --------------------------------------------------
            // CONFIDENCE
            // --------------------------------------------------
            Center(
              child: Text(
                "Prediction Confidence: "
                "${(confidence * 100).toStringAsFixed(2)}%",
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),

            const SizedBox(height: 12),

            Text(
              isMalnourished
                  ? "The model has identified a "
                    "potential malnutrition condition."
                  : "The model has not identified "
                    "malnutrition.",
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 15,
              ),
            ),

            const SizedBox(height: 25),

            // --------------------------------------------------
            // RISK LEVEL
            // --------------------------------------------------
            Card(
              child: Padding(
                padding: const EdgeInsets.all(18),

                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment:
                          MainAxisAlignment.center,
                      children: [
                        Icon(
                          riskIcon,
                          color: riskColor,
                          size: 28,
                        ),

                        const SizedBox(width: 10),

                        const Text(
                          "Risk Level",
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 10),

                    Text(
                      riskLevel,
                      style: TextStyle(
                        color: riskColor,
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 8),

                    Text(
                      riskMessage,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 30),

            // --------------------------------------------------
            // PROBABILITIES
            // --------------------------------------------------
            const Text(
              "Prediction Probabilities",
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 15),

            if (probabilities.isEmpty)
              const Text(
                "Probability information "
                "was not available.",
              )
            else
              ...List.generate(
                probabilities.length,
                (index) {
                  final probability =
                      probabilities[index];

                  final percentage =
                      probability * 100;

                  return Card(
                    child: Padding(
                      padding:
                          const EdgeInsets.all(14),

                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              className(index),
                              style:
                                  const TextStyle(
                                fontSize: 16,
                                fontWeight:
                                    FontWeight.w600,
                              ),
                            ),
                          ),

                          Text(
                            "${percentage.toStringAsFixed(2)}%",
                            style:
                                const TextStyle(
                              fontSize: 16,
                              fontWeight:
                                  FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),

            const SizedBox(height: 30),

            // --------------------------------------------------
            // IMPORTANT NOTICE
            // --------------------------------------------------
            Card(
              child: Padding(
                padding:
                    const EdgeInsets.all(14),

                child: const Text(
                  "This prediction is generated by "
                  "a machine-learning model and should "
                  "be used as a screening aid, not as "
                  "a medical diagnosis.",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 12,
                  ),
                ),
              ),
            ),

            const SizedBox(height: 25),

            // --------------------------------------------------
            // BACK TO HOME
            // --------------------------------------------------
            SizedBox(
              width: double.infinity,

              child: ElevatedButton(
                onPressed: () {
                  Navigator.popUntil(
                    context,
                    (route) => route.isFirst,
                  );
                },
                child: const Text(
                  "Back to Home",
                ),
              ),
            ),

            const SizedBox(height: 10),
          ],
        ),
      ),
    );
  }
}