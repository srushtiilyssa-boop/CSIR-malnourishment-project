import 'package:flutter/material.dart';
import 'theme/app_theme.dart';
import 'screens/home_screen.dart';

class MalnutritionApp extends StatelessWidget {
  const MalnutritionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: "Child Nutrition AI",
      theme: AppTheme.lightTheme,
      home: const HomeScreen(),
    );
  }
}