import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../theme/app_text_styles.dart';
import '../widgets/app_button.dart';
import 'mother_screen.dart';

class HealthScreen extends StatefulWidget {
  final Patient patient;

  const HealthScreen({
    super.key,
    required this.patient,
  });

  @override
  State<HealthScreen> createState() =>
      _HealthScreenState();
}

class _HealthScreenState
    extends State<HealthScreen> {
  bool breastfeeding = false;
  bool diarrhea = false;
  bool fever = false;
  bool cough = false;

  @override
  void initState() {
    super.initState();

    breastfeeding =
        widget.patient.breastfeeding == 1;

    diarrhea =
        widget.patient.diarrhea == 1;

    fever =
        widget.patient.fever == 1;

    cough =
        widget.patient.cough == 1;
  }

  void saveData() {
    // Convert user-friendly Yes/No switches
    // into the numeric values required
    // by the ML model.
    widget.patient.breastfeeding =
        breastfeeding ? 1 : 0;

    widget.patient.diarrhea =
        diarrhea ? 1 : 0;

    widget.patient.fever =
        fever ? 1 : 0;

    widget.patient.cough =
        cough ? 1 : 0;
  }

  Widget buildHealthSwitch({
    required String title,
    required String description,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Card(
      margin: const EdgeInsets.only(
        bottom: 14,
      ),
      child: SwitchListTile(
        title: Text(
          title,
          style: AppTextStyles.title,
        ),
        subtitle: Text(
          description,
          style: AppTextStyles.subtitle,
        ),
        value: value,
        onChanged: onChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Step 2 of 4",
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              "Health Information",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 8),

            const Text(
              "Answer the following questions "
              "about the child's current health.",
            ),

            const SizedBox(height: 25),

            // BREASTFEEDING
            buildHealthSwitch(
              title: "Breastfeeding",
              description:
                  "Is the child currently breastfeeding?",
              value: breastfeeding,
              onChanged: (value) {
                setState(() {
                  breastfeeding = value;
                });
              },
            ),

            // DIARRHEA
            buildHealthSwitch(
              title: "Diarrhea",
              description:
                  "Has the child recently had diarrhea?",
              value: diarrhea,
              onChanged: (value) {
                setState(() {
                  diarrhea = value;
                });
              },
            ),

            // FEVER
            buildHealthSwitch(
              title: "Fever",
              description:
                  "Has the child recently had a fever?",
              value: fever,
              onChanged: (value) {
                setState(() {
                  fever = value;
                });
              },
            ),

            // COUGH
            buildHealthSwitch(
              title: "Cough",
              description:
                  "Has the child recently had a cough?",
              value: cough,
              onChanged: (value) {
                setState(() {
                  cough = value;
                });
              },
            ),

            const SizedBox(height: 25),

            AppButton(
              text: "Next",
              onPressed: () {
                try {
                  saveData();

                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) =>
                          MotherScreen(
                        patient:
                            widget.patient,
                      ),
                    ),
                  );
                } catch (e) {
                  ScaffoldMessenger.of(
                    context,
                  ).showSnackBar(
                    SnackBar(
                      content: Text(
                        e.toString(),
                      ),
                    ),
                  );
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}