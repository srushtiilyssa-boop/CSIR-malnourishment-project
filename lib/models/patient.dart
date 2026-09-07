class Patient {
  // ============================================================
  // CHILD INFORMATION
  // ============================================================

  int childAgeMonths;

  String gender;

  // IMPORTANT:
  // This is the DATASET CODE.
  // Do not treat this as kilograms.
  //
  // Valid dataset values:
  // 1, 2, 3, 4, 5, 8
  double birthWeight;

  double weightKg;
  double heightCm;

  // ============================================================
  // HEALTH INFORMATION
  // ============================================================

  // IMPORTANT:
  // This is the DATASET breastfeeding code.
  //
  // The dataset contains numeric duration/codes including
  // 0-... and special codes such as 95, 98.
  int breastfeeding;

  // Dataset uses 0 / 1.
  int diarrhea;
  int fever;
  int cough;

  // ============================================================
  // MOTHER INFORMATION
  // ============================================================

  int motherAge;

  // Dataset:
  // 0 = No Education
  // 1 = Primary
  // 2 = Secondary
  // 3 = Higher
  int motherEducation;

  double motherBmi;

  int childrenEverBorn;
  int birthOrder;

  int antenatalVisits;

  // ============================================================
  // HOUSEHOLD INFORMATION
  // ============================================================

  // Dataset:
  // 1 = Urban
  // 2 = Rural
  int urbanRural;

  // Dataset:
  // 1 = Poorest
  // 2 = Poorer
  // 3 = Middle
  // 4 = Richer
  // 5 = Richest
  int wealthIndex;

  // IMPORTANT:
  // These are raw dataset codes.
  // Do NOT convert them into artificial 0/1/2/3 categories.
  int waterSource;

  int timeToWater;

  int sanitationToiletFacility;

  int handwashingFacility;

  int householdMembers;

  // ============================================================
  // CONSTRUCTOR
  // ============================================================

  Patient({
    this.childAgeMonths = 0,
    this.gender = "Male",
    this.birthWeight = 0.0,
    this.weightKg = 0.0,
    this.heightCm = 0.0,

    this.breastfeeding = 0,
    this.diarrhea = 0,
    this.fever = 0,
    this.cough = 0,

    this.motherAge = 0,
    this.motherEducation = 0,
    this.motherBmi = 0.0,
    this.childrenEverBorn = 0,
    this.birthOrder = 0,
    this.antenatalVisits = 0,

    this.urbanRural = 1,
    this.wealthIndex = 3,
    this.waterSource = 12,
    this.timeToWater = 0,
    this.sanitationToiletFacility = 44,
    this.handwashingFacility = 34,
    this.householdMembers = 0,
  });

  // ============================================================
  // EXACT DATASET CODE TABLES
  // ============================================================

  static const List<int> validBreastfeedingCodes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    95,
    98,
  ];

  static const List<int> validWaterSourceCodes = [
    11,
    12,
    13,
    14,
    21,
    31,
    32,
    41,
    42,
    43,
    51,
    61,
    62,
    71,
    92,
    96,
    97,
  ];

  static const List<int> validToiletCodes = [
    11,
    12,
    13,
    14,
    15,
    21,
    22,
    23,
    31,
    41,
    44,
    96,
    97,
  ];

  static const List<int> validHandwashingCodes = [
    11,
    12,
    13,
    21,
    22,
    23,
    24,
    31,
    32,
    33,
    34,
    35,
    36,
    96,
    97,
  ];

  // ============================================================
  // CHILD AGE IN YEARS
  // ============================================================
  //
  // The dataset contains decimal years.
  //
  // Example:
  // 12 months → 1.0
  // 24 months → 2.0
  // 25 months → 2.1
  //
  // We MUST NOT use integer division.
  // ============================================================

  double get childAgeYears {
    return double.parse(
      (childAgeMonths / 12.0).toStringAsFixed(1),
    );
  }

  // ============================================================
  // VALIDATION
  // ============================================================

  void validateForModel() {
    if (childAgeMonths < 0 || childAgeMonths > 59) {
      throw Exception(
        "child_age_months must be between 0 and 59.",
      );
    }

    if (gender != "Male" && gender != "Female") {
      throw Exception(
        "gender must be Male or Female.",
      );
    }

    if (birthWeight < 0.5 ||
        birthWeight > 6.0) {
      throw Exception(
        "birth_weight must be between 0.5 and 6.0 kg.",
      );
    }

    if (weightKg < 1.7 || weightKg > 29.8) {
      throw Exception(
        "weight_kg is outside the dataset range.",
      );
    }

    if (heightCm < 45.0 || heightCm > 120.0) {
      throw Exception(
        "height_cm is outside the dataset range.",
      );
    }

    if (!validBreastfeedingCodes.contains(breastfeeding)) {
      throw Exception(
        "Invalid breastfeeding dataset code: "
        "$breastfeeding",
      );
    }

    _validateBinary("diarrhea", diarrhea);
    _validateBinary("fever", fever);
    _validateBinary("cough", cough);

    if (motherAge < 15 || motherAge > 49) {
      throw Exception(
        "mother_age must be between 15 and 49.",
      );
    }

    if (motherEducation < 0 ||
        motherEducation > 3) {
      throw Exception(
        "mother_education must be 0, 1, 2 or 3.",
      );
    }

    if (motherBmi < 12.02 ||
        motherBmi > 59.99) {
      throw Exception(
        "mother_bmi is outside the dataset range.",
      );
    }

    if (childrenEverBorn < 1 ||
        childrenEverBorn > 16) {
      throw Exception(
        "children_ever_born is outside the "
        "dataset range.",
      );
    }

    if (birthOrder < 1 || birthOrder > 6) {
      throw Exception(
        "birth_order must be between 1 and 6.",
      );
    }

    if (urbanRural != 1 &&
        urbanRural != 2) {
      throw Exception(
        "urban_rural must be 1 or 2.",
      );
    }

    if (wealthIndex < 1 ||
        wealthIndex > 5) {
      throw Exception(
        "wealth_index must be between 1 and 5.",
      );
    }

    if (!validWaterSourceCodes
        .contains(waterSource)) {
      throw Exception(
        "Invalid water_source dataset code: "
        "$waterSource",
      );
    }

    if (timeToWater < 0 ||
        timeToWater > 900) {
      throw Exception(
        "time_to_water must be between 0 and 900.",
      );
    }

    if (!validToiletCodes
        .contains(sanitationToiletFacility)) {
      throw Exception(
        "Invalid sanitation_toilet_facility "
        "dataset code: "
        "$sanitationToiletFacility",
      );
    }

    if (!validHandwashingCodes
        .contains(handwashingFacility)) {
      throw Exception(
        "Invalid handwashing_facility "
        "dataset code: "
        "$handwashingFacility",
      );
    }

    if (householdMembers < 2 ||
        householdMembers > 35) {
      throw Exception(
        "household_members must be between 2 and 35.",
      );
    }

    if (antenatalVisits < 0 ||
        antenatalVisits > 30) {
      throw Exception(
        "antenatal_visits must be between 0 and 30.",
      );
    }
  }

  void _validateBinary(
    String name,
    int value,
  ) {
    if (value != 0 && value != 1) {
      throw Exception(
        "$name must be 0 or 1.",
      );
    }
  }

  // ============================================================
  // EXACT 23-FEATURE VECTOR
  // ============================================================

  List<double> toFeatureVector() {
    validateForModel();

    return [
      // 1
      childAgeMonths.toDouble(),

      // 2
      // Dataset:
      // Male = 1
      // Female = 2
      gender == "Male" ? 1.0 : 2.0,

      // 3
      birthWeight.toDouble(),

      // 4
      weightKg,

      // 5
      heightCm,

      // 6
      breastfeeding.toDouble(),

      // 7
      diarrhea.toDouble(),

      // 8
      fever.toDouble(),

      // 9
      cough.toDouble(),

      // 10
      motherAge.toDouble(),

      // 11
      motherEducation.toDouble(),

      // 12
      motherBmi,

      // 13
      childrenEverBorn.toDouble(),

      // 14
      birthOrder.toDouble(),

      // 15
      urbanRural.toDouble(),

      // 16
      wealthIndex.toDouble(),

      // 17
      waterSource.toDouble(),

      // 18
      timeToWater.toDouble(),

      // 19
      sanitationToiletFacility.toDouble(),

      // 20
      handwashingFacility.toDouble(),

      // 21
      childAgeYears,

      // 22
      householdMembers.toDouble(),

      // 23
      antenatalVisits.toDouble(),
    ];
  }

  // ============================================================
  // DEBUGGING
  // ============================================================

  void printFeatureVector() {
    final features = toFeatureVector();

    print("");
    print("==============================================");
    print("EXACT DATASET FEATURE VECTOR");
    print("==============================================");

    const names = [
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

    for (int i = 0; i < features.length; i++) {
      print(
        "${i + 1}. ${names[i]} = ${features[i]}",
      );
    }

    print("----------------------------------------------");
    print("TOTAL FEATURES: ${features.length}");
    print("==============================================");
  }
}