import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../widgets/app_button.dart';
import '../widgets/app_dropdown.dart';
import '../widgets/app_textfield.dart';
import 'household_screen.dart';

class MotherScreen extends StatefulWidget {
  final Patient patient;

  const MotherScreen({
    super.key,
    required this.patient,
  });

  @override
  State<MotherScreen> createState() =>
      _MotherScreenState();
}

class _MotherScreenState
    extends State<MotherScreen> {
  final _formKey =
      GlobalKey<FormState>();

  late TextEditingController
      motherAgeController;

  late TextEditingController
      bmiController;

  late TextEditingController
      childrenBornController;

  late TextEditingController
      birthOrderController;

  late TextEditingController
      antenatalController;

  String education =
      "No Education";

  @override
  void initState() {
    super.initState();

    motherAgeController =
        TextEditingController(
      text:
          widget.patient.motherAge == 0
              ? ""
              : widget.patient
                  .motherAge
                  .toString(),
    );

    bmiController =
        TextEditingController(
      text:
          widget.patient.motherBmi == 0
              ? ""
              : widget.patient
                  .motherBmi
                  .toString(),
    );

    childrenBornController =
        TextEditingController(
      text:
          widget.patient
                  .childrenEverBorn ==
              0
          ? ""
          : widget.patient
              .childrenEverBorn
              .toString(),
    );

    birthOrderController =
        TextEditingController(
      text:
          widget.patient.birthOrder ==
                  0
              ? ""
              : widget.patient
                  .birthOrder
                  .toString(),
    );

    antenatalController =
        TextEditingController(
      text:
          widget.patient
                  .antenatalVisits ==
              0
          ? ""
          : widget.patient
              .antenatalVisits
              .toString(),
    );

    education =
        _educationLabel(
      widget.patient.motherEducation,
    );
  }

  @override
  void dispose() {
    motherAgeController.dispose();
    bmiController.dispose();
    childrenBornController.dispose();
    birthOrderController.dispose();
    antenatalController.dispose();

    super.dispose();
  }

  void saveData() {
    final age =
        int.tryParse(
      motherAgeController.text.trim(),
    );

    final bmi =
        double.tryParse(
      bmiController.text.trim(),
    );

    final children =
        int.tryParse(
      childrenBornController.text.trim(),
    );

    final order =
        int.tryParse(
      birthOrderController.text.trim(),
    );

    final antenatal =
        int.tryParse(
      antenatalController.text.trim(),
    );

    if (age == null ||
        age < 15 ||
        age > 49) {
      throw Exception(
        "Mother age must be 15-49.",
      );
    }

    if (bmi == null ||
        bmi < 12.02 ||
        bmi > 59.99) {
      throw Exception(
        "Mother BMI is outside "
        "the dataset range.",
      );
    }

    if (children == null ||
        children < 1 ||
        children > 16) {
      throw Exception(
        "Children ever born must be 1-16.",
      );
    }

    if (order == null ||
        order < 1 ||
        order > 6) {
      throw Exception(
        "Birth order must be 1-6.",
      );
    }

    if (antenatal == null ||
        antenatal < 0 ||
        antenatal > 30) {
      throw Exception(
        "Antenatal visits must be 0-30.",
      );
    }

    widget.patient.motherAge =
        age;

    widget.patient.motherEducation =
        _educationCode(
      education,
    );

    widget.patient.motherBmi =
        bmi;

    widget.patient.childrenEverBorn =
        children;

    widget.patient.birthOrder =
        order;

    widget.patient.antenatalVisits =
        antenatal;
  }

  int _educationCode(
    String value,
  ) {
    switch (value) {
      case "No Education":
        return 0;
      case "Primary":
        return 1;
      case "Secondary":
        return 2;
      case "Higher":
        return 3;
      default:
        throw Exception(
          "Invalid education value.",
        );
    }
  }

  String _educationLabel(
    int code,
  ) {
    switch (code) {
      case 0:
        return "No Education";
      case 1:
        return "Primary";
      case 2:
        return "Secondary";
      case 3:
        return "Higher";
      default:
        return "No Education";
    }
  }

  @override
  Widget build(
    BuildContext context,
  ) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Step 3 of 4",
        ),
      ),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding:
                const EdgeInsets.all(20),
            children: [
              const Text(
                "Mother Information",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              const SizedBox(height: 25),

              AppTextField(
                controller:
                    motherAgeController,
                label: "Mother Age",
                icon: Icons.person,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 15),

              AppDropdown(
                label: "Education",
                icon: Icons.school,
                value: education,
                items: const [
                  "No Education",
                  "Primary",
                  "Secondary",
                  "Higher",
                ],
                onChanged: (value) {
                  setState(() {
                    education = value!;
                  });
                },
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    bmiController,
                label: "Mother BMI",
                icon:
                    Icons.monitor_weight,
                keyboardType:
                    const TextInputType
                        .numberWithOptions(
                  decimal: true,
                ),
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    childrenBornController,
                label:
                    "Children Ever Born",
                icon: Icons.child_care,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    birthOrderController,
                label: "Birth Order",
                icon:
                    Icons.format_list_numbered,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    antenatalController,
                label:
                    "Antenatal Visits",
                icon:
                    Icons.local_hospital,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 30),

              AppButton(
                text: "Next",
                onPressed: () {
                  if (!_formKey
                      .currentState!
                      .validate()) {
                    return;
                  }

                  try {
                    saveData();

                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) =>
                            HouseholdScreen(
                          patient:
                              widget.patient,
                        ),
                      ),
                    );
                  } catch (e) {
                    ScaffoldMessenger
                        .of(context)
                        .showSnackBar(
                      SnackBar(
                        content:
                            Text(
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
      ),
    );
  }
}