import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../widgets/app_button.dart';
import '../widgets/app_dropdown.dart';
import '../widgets/app_textfield.dart';
import 'prediction_screen.dart';

class HouseholdScreen extends StatefulWidget {
  final Patient patient;

  const HouseholdScreen({
    super.key,
    required this.patient,
  });

  @override
  State<HouseholdScreen> createState() =>
      _HouseholdScreenState();
}

class _HouseholdScreenState
    extends State<HouseholdScreen> {
  final _formKey =
      GlobalKey<FormState>();

  late TextEditingController
      timeToWaterController;

  late TextEditingController
      householdMembersController;

  late TextEditingController
      waterSourceController;

  late TextEditingController
      toiletController;

  late TextEditingController
      handwashingController;

  int urbanRural = 1;
  int wealthIndex = 3;

  @override
  void initState() {
    super.initState();

    timeToWaterController =
        TextEditingController(
      text:
          widget.patient.timeToWater
              .toString(),
    );

    householdMembersController =
        TextEditingController(
      text:
          widget.patient.householdMembers
              .toString(),
    );

    waterSourceController =
        TextEditingController(
      text:
          widget.patient.waterSource
              .toString(),
    );

    toiletController =
        TextEditingController(
      text:
          widget.patient
              .sanitationToiletFacility
              .toString(),
    );

    handwashingController =
        TextEditingController(
      text:
          widget.patient
              .handwashingFacility
              .toString(),
    );

    urbanRural =
        widget.patient.urbanRural;

    wealthIndex =
        widget.patient.wealthIndex;
  }

  @override
  void dispose() {
    timeToWaterController.dispose();
    householdMembersController.dispose();
    waterSourceController.dispose();
    toiletController.dispose();
    handwashingController.dispose();

    super.dispose();
  }

  void saveData() {
    final water =
        int.tryParse(
      waterSourceController.text.trim(),
    );

    final toilet =
        int.tryParse(
      toiletController.text.trim(),
    );

    final handwash =
        int.tryParse(
      handwashingController.text.trim(),
    );

    final time =
        int.tryParse(
      timeToWaterController.text.trim(),
    );

    final members =
        int.tryParse(
      householdMembersController.text.trim(),
    );

    if (water == null ||
        !Patient.validWaterSourceCodes
            .contains(water)) {
      throw Exception(
        "Invalid water_source dataset code.",
      );
    }

    if (toilet == null ||
        !Patient.validToiletCodes
            .contains(toilet)) {
      throw Exception(
        "Invalid sanitation/toilet "
        "dataset code.",
      );
    }

    if (handwash == null ||
        !Patient.validHandwashingCodes
            .contains(handwash)) {
      throw Exception(
        "Invalid handwashing dataset code.",
      );
    }

    if (time == null ||
        time < 0 ||
        time > 900) {
      throw Exception(
        "Time to water must be 0-900.",
      );
    }

    if (members == null ||
        members < 2 ||
        members > 35) {
      throw Exception(
        "Household members must be 2-35.",
      );
    }

    widget.patient.urbanRural =
        urbanRural;

    widget.patient.wealthIndex =
        wealthIndex;

    widget.patient.waterSource =
        water;

    widget.patient
            .sanitationToiletFacility =
        toilet;

    widget.patient
            .handwashingFacility =
        handwash;

    widget.patient.timeToWater =
        time;

    widget.patient.householdMembers =
        members;
  }

  @override
  Widget build(
    BuildContext context,
  ) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "Step 4 of 4",
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
                "Household Information",
                style: TextStyle(
                  fontSize: 24,
                  fontWeight:
                      FontWeight.bold,
                ),
              ),

              const SizedBox(height: 8),

              const Text(
                "Values below are sent using "
                "the exact dataset coding.",
              ),

              const SizedBox(height: 25),

              // ==================================================
              // URBAN / RURAL
              // ==================================================

              AppDropdown(
                label: "Location",
                icon:
                    Icons.location_city,
                value: urbanRural == 1
                    ? "Urban"
                    : "Rural",
                items: const [
                  "Urban",
                  "Rural",
                ],
                onChanged: (value) {
                  setState(() {
                    urbanRural =
                        value == "Urban"
                            ? 1
                            : 2;
                  });
                },
              ),

              const SizedBox(height: 15),

              // ==================================================
              // WEALTH
              // ==================================================

              AppDropdown(
                label: "Wealth Index",
                icon:
                    Icons.account_balance_wallet,
                value:
                    _wealthLabel(
                  wealthIndex,
                ),
                items: const [
                  "Poorest",
                  "Poorer",
                  "Middle",
                  "Richer",
                  "Richest",
                ],
                onChanged: (value) {
                  setState(() {
                    wealthIndex =
                        _wealthCode(
                      value!,
                    );
                  });
                },
              ),

              const SizedBox(height: 20),

              // ==================================================
              // RAW DATASET CODES
              // ==================================================

              AppTextField(
                controller:
                    waterSourceController,
                label:
                    "Water Source Dataset Code",
                icon:
                    Icons.water_drop,
                keyboardType:
                    TextInputType.number,
                validator: (value) {
                  final code =
                      int.tryParse(
                    value ?? "",
                  );

                  if (code == null ||
                      !Patient
                          .validWaterSourceCodes
                          .contains(code)) {
                    return "Invalid dataset code";
                  }

                  return null;
                },
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    toiletController,
                label:
                    "Toilet Facility Dataset Code",
                icon: Icons.wc,
                keyboardType:
                    TextInputType.number,
                validator: (value) {
                  final code =
                      int.tryParse(
                    value ?? "",
                  );

                  if (code == null ||
                      !Patient
                          .validToiletCodes
                          .contains(code)) {
                    return "Invalid dataset code";
                  }

                  return null;
                },
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    handwashingController,
                label:
                    "Handwashing Dataset Code",
                icon:
                    Icons.clean_hands,
                keyboardType:
                    TextInputType.number,
                validator: (value) {
                  final code =
                      int.tryParse(
                    value ?? "",
                  );

                  if (code == null ||
                      !Patient
                          .validHandwashingCodes
                          .contains(code)) {
                    return "Invalid dataset code";
                  }

                  return null;
                },
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    timeToWaterController,
                label:
                    "Time to Water (minutes)",
                icon: Icons.timer,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 15),

              AppTextField(
                controller:
                    householdMembersController,
                label:
                    "Household Members",
                icon:
                    Icons.family_restroom,
                keyboardType:
                    TextInputType.number,
              ),

              const SizedBox(height: 30),

              AppButton(
                text:
                    "Run AI Prediction",
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
                            PredictionScreen(
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

  String _wealthLabel(
    int code,
  ) {
    switch (code) {
      case 1:
        return "Poorest";
      case 2:
        return "Poorer";
      case 3:
        return "Middle";
      case 4:
        return "Richer";
      case 5:
        return "Richest";
      default:
        return "Middle";
    }
  }

  int _wealthCode(
    String value,
  ) {
    switch (value) {
      case "Poorest":
        return 1;
      case "Poorer":
        return 2;
      case "Middle":
        return 3;
      case "Richer":
        return 4;
      case "Richest":
        return 5;
      default:
        throw Exception(
          "Invalid wealth index.",
        );
    }
  }
}