import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../widgets/app_button.dart';
import '../widgets/app_dropdown.dart';
import '../widgets/app_textfield.dart';
import 'health_screen.dart';

class ChildInfoScreen extends StatefulWidget {
  final Patient patient;

  const ChildInfoScreen({
    super.key,
    required this.patient,
  });

  @override
  State<ChildInfoScreen> createState() =>
      _ChildInfoScreenState();
}

class _ChildInfoScreenState
    extends State<ChildInfoScreen> {
  final _formKey = GlobalKey<FormState>();

  late TextEditingController ageController;
  late TextEditingController birthWeightController;
  late TextEditingController weightController;
  late TextEditingController heightController;

  String gender = "Male";

  @override
  void initState() {
    super.initState();

    ageController = TextEditingController(
      text: widget.patient.childAgeMonths == 0
          ? ""
          : widget.patient.childAgeMonths.toString(),
    );

    birthWeightController = TextEditingController(
      text: widget.patient.birthWeight == 0
          ? ""
          : widget.patient.birthWeight.toString(),
    );

    weightController = TextEditingController(
      text: widget.patient.weightKg == 0
          ? ""
          : widget.patient.weightKg.toString(),
    );

    heightController = TextEditingController(
      text: widget.patient.heightCm == 0
          ? ""
          : widget.patient.heightCm.toString(),
    );

    gender = widget.patient.gender;
  }

  @override
  void dispose() {
    ageController.dispose();
    birthWeightController.dispose();
    weightController.dispose();
    heightController.dispose();

    super.dispose();
  }

  void saveData() {
    final age = int.tryParse(
      ageController.text.trim(),
    );

    final birthWeight = double.tryParse(
      birthWeightController.text.trim(),
    );

    final weight = double.tryParse(
      weightController.text.trim(),
    );

    final height = double.tryParse(
      heightController.text.trim(),
    );

    // -----------------------------
    // AGE
    // -----------------------------
    if (age == null || age < 0 || age > 59) {
      throw Exception(
        "Age must be between 0 and 59 months.",
      );
    }

    // -----------------------------
    // BIRTH WEIGHT
    // -----------------------------
    if (birthWeight == null ||
        birthWeight < 0.5 ||
        birthWeight > 6.0) {
      throw Exception(
        "Birth weight must be between "
        "0.5 and 6.0 kg.",
      );
    }

    // -----------------------------
    // CURRENT WEIGHT
    // -----------------------------
    if (weight == null ||
        weight < 1.7 ||
        weight > 29.8) {
      throw Exception(
        "Current weight must be between "
        "1.7 and 29.8 kg.",
      );
    }

    // -----------------------------
    // CURRENT HEIGHT
    // -----------------------------
    if (height == null ||
        height < 45.0 ||
        height > 120.0) {
      throw Exception(
        "Current height must be between "
        "45 and 120 cm.",
      );
    }

    // -----------------------------
    // SAVE
    // -----------------------------
    widget.patient.childAgeMonths = age;
    widget.patient.gender = gender;

    // Actual birth weight in kilograms.
    widget.patient.birthWeight = birthWeight;

    widget.patient.weightKg = weight;
    widget.patient.heightCm = height;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Step 1 of 4",
        ),
      ),

      body: SafeArea(
        child: Form(
          key: _formKey,

          child: ListView(
            padding: const EdgeInsets.all(20),

            children: [
              const Text(
                "Child Information",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 8),

              const Text(
                "Enter the child's basic information "
                "using the units shown below.",
              ),

              const SizedBox(height: 25),

              // AGE
              AppTextField(
                controller: ageController,
                label: "Child Age (Months)",
                icon: Icons.calendar_month,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 5),

              const Text(
                "Enter age from 0 to 59 months. "
                "Example: 24 for a 2-year-old child.",
                style: TextStyle(
                  fontSize: 12,
                ),
              ),

              const SizedBox(height: 15),

              // GENDER
              AppDropdown(
                label: "Gender",
                icon: Icons.person,
                value: gender,
                items: const [
                  "Male",
                  "Female",
                ],
                onChanged: (value) {
                  if (value == null) return;

                  setState(() {
                    gender = value;
                  });
                },
              ),

              // BIRTH WEIGHT
              AppTextField(
                controller:
                    birthWeightController,
                label: "Birth Weight (kg)",
                icon: Icons.child_friendly,
                keyboardType:
                    const TextInputType.numberWithOptions(
                  decimal: true,
                ),
              ),

              const SizedBox(height: 5),

              const Text(
                "Enter the child's birth weight "
                "in kilograms. Valid range: 0.5–6.0 kg. "
                "Example: 2.8",
                style: TextStyle(
                  fontSize: 12,
                ),
              ),

              const SizedBox(height: 15),

              // CURRENT WEIGHT
              AppTextField(
                controller: weightController,
                label: "Current Weight (kg)",
                icon: Icons.monitor_weight,
                keyboardType:
                    const TextInputType.numberWithOptions(
                  decimal: true,
                ),
              ),

              const SizedBox(height: 5),

              const Text(
                "Enter the child's current weight "
                "in kilograms. Example: 12.5",
                style: TextStyle(
                  fontSize: 12,
                ),
              ),

              const SizedBox(height: 15),

              // CURRENT HEIGHT
              AppTextField(
                controller: heightController,
                label: "Current Height (cm)",
                icon: Icons.height,
                keyboardType:
                    const TextInputType.numberWithOptions(
                  decimal: true,
                ),
              ),

              const SizedBox(height: 5),

              const Text(
                "Enter the child's current height "
                "in centimetres. Example: 85.5",
                style: TextStyle(
                  fontSize: 12,
                ),
              ),

              const SizedBox(height: 30),

              AppButton(
                text: "Next",

                onPressed: () {
                  if (!_formKey.currentState!
                      .validate()) {
                    return;
                  }

                  try {
                    saveData();

                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) =>
                            HealthScreen(
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
                          e.toString().replaceFirst(
                            "Exception: ",
                            "",
                          ),
                        ),
                      ),
                    );
                  }
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}