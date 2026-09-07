import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import 'app_colors.dart';

class AppTextStyles {
  AppTextStyles._();

  static TextStyle heading = GoogleFonts.poppins(
    fontSize: 30,
    fontWeight: FontWeight.bold,
    color: AppColors.textDark,
  );

  static TextStyle title = GoogleFonts.poppins(
    fontSize: 22,
    fontWeight: FontWeight.w600,
    color: AppColors.textDark,
  );

  static TextStyle subtitle = GoogleFonts.poppins(
    fontSize: 16,
    color: AppColors.textLight,
  );

  static TextStyle body = GoogleFonts.poppins(
    fontSize: 15,
    color: AppColors.textDark,
  );

  static TextStyle button = GoogleFonts.poppins(
    fontWeight: FontWeight.w600,
    fontSize: 18,
    color: Colors.white,
  );
}