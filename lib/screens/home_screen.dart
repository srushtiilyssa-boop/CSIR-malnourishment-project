import 'package:flutter/material.dart';

import '../models/patient.dart';
import '../theme/app_colors.dart';
import '../theme/app_spacing.dart';
import '../theme/app_text_styles.dart';
import 'child_info_screen.dart';
import 'history_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Widget _overviewCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
  }) {
    return Expanded(
      child: Container(
        height: 165,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 15,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 58,
              height: 58,
              decoration: BoxDecoration(
                color: color.withOpacity(0.12),
                shape: BoxShape.circle,
              ),
              child: Icon(
                icon,
                color: color,
                size: 27,
              ),
            ),
            const SizedBox(height: 14),
            Text(
              title,
              textAlign: TextAlign.center,
              style: AppTextStyles.title.copyWith(fontSize: 16),
            ),
            const SizedBox(height: 5),
            Text(
              subtitle,
              textAlign: TextAlign.center,
              style: AppTextStyles.subtitle.copyWith(fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }

  Widget _aboutItem({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(
              icon,
              color: color,
              size: 22,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: AppTextStyles.title.copyWith(
                    fontSize: 15,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  description,
                  style: AppTextStyles.subtitle.copyWith(
                    fontSize: 12.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _openAbout(BuildContext context) {
    showDialog<void>(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          title: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(9),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(
                  Icons.info_outline_rounded,
                  color: AppColors.primary,
                ),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Text('About this screening'),
              ),
            ],
          ),
          content: SizedBox(
            width: 650,
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Why use this screening?',
                    style: AppTextStyles.title.copyWith(
                      fontSize: 19,
                    ),
                  ),
                  const SizedBox(height: 14),

                  _aboutItem(
                    icon: Icons.psychology_rounded,
                    title: 'AI Powered',
                    description:
                        'Intelligent prediction using the installed AI screening model.',
                    color: AppColors.secondary,
                  ),

                  _aboutItem(
                    icon: Icons.monitor_heart_rounded,
                    title: 'Growth',
                    description:
                        'Uses child health and growth information to support the assessment.',
                    color: AppColors.accent,
                  ),

                  _aboutItem(
                    icon: Icons.lock_rounded,
                    title: 'Private',
                    description:
                        'Patient information remains on the device and is not sent to an external server by the screening process.',
                    color: AppColors.lavender,
                  ),

                  const SizedBox(height: 14),

                  Text(
                    'Assessment features',
                    style: AppTextStyles.title.copyWith(
                      fontSize: 19,
                    ),
                  ),

                  const SizedBox(height: 14),

                  _aboutItem(
                    icon: Icons.analytics_rounded,
                    title: 'Nutrition Risk Assessment',
                    description:
                        'Evaluates the entered child health information and produces a model prediction.',
                    color: AppColors.primary,
                  ),

                  _aboutItem(
                    icon: Icons.speed_rounded,
                    title: 'Quick Screening',
                    description:
                        'A simple step-by-step form makes child information entry straightforward.',
                    color: AppColors.secondary,
                  ),

                  _aboutItem(
                    icon: Icons.shield_rounded,
                    title: 'Offline & Private',
                    description:
                        'The prediction model runs locally on the device.',
                    color: AppColors.accent,
                  ),
                ],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: const Text('Close'),
            ),
          ],
        );
      },
    );
  }

  void _startScreening(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => ChildInfoScreen(
          patient: Patient(),
        ),
      ),
    );
  }

  void _openHistory(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => const HistoryScreen(),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppSpacing.lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // HEADER
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(0.14),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: const Icon(
                      Icons.health_and_safety_rounded,
                      color: AppColors.primary,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Child Nutrition AI',
                          style: AppTextStyles.title,
                        ),
                        Text(
                          'Child health screening assistant',
                          style: AppTextStyles.subtitle.copyWith(
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 11,
                      vertical: 7,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.secondary.withOpacity(0.16),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Row(
                      children: [
                        Icon(
                          Icons.cloud_off_rounded,
                          size: 15,
                          color: AppColors.textDark,
                        ),
                        SizedBox(width: 5),
                        Text(
                          'Offline',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 28),

              Text(
                'Start a child nutrition screening',
                style: AppTextStyles.heading.copyWith(
                  fontSize: 25,
                ),
              ),

              const SizedBox(height: 7),

              Text(
                'Enter the child information to begin an assessment.',
                style: AppTextStyles.subtitle,
              ),

              const SizedBox(height: 18),

              // MAIN SCREENING CARD
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(24),
                  gradient: const LinearGradient(
                    colors: [
                      AppColors.primary,
                      AppColors.lavender,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: AppColors.primary.withOpacity(0.20),
                      blurRadius: 18,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(13),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.24),
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.child_care_rounded,
                            color: Colors.white,
                            size: 30,
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Text(
                            'New Child Assessment',
                            style: AppTextStyles.heading.copyWith(
                              color: Colors.white,
                              fontSize: 21,
                            ),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 18),

                    Text(
                      'Complete the screening form and let the local AI model assess the entered information.',
                      style: AppTextStyles.subtitle.copyWith(
                        color: Colors.white.withOpacity(0.9),
                      ),
                    ),

                    const SizedBox(height: 20),

                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: () => _startScreening(context),
                        icon: const Icon(
                          Icons.arrow_forward_rounded,
                        ),
                        label: const Text(
                          'START SCREENING',
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),

              // SMALL OVERVIEW
              Text(
                'Quick overview',
                style: AppTextStyles.title.copyWith(
                  fontSize: 19,
                ),
              ),

              const SizedBox(height: 12),

              Row(
                children: [
                  _overviewCard(
                    icon: Icons.psychology_rounded,
                    title: 'AI Powered',
                    subtitle: 'Intelligent prediction',
                    color: AppColors.secondary,
                  ),
                  const SizedBox(width: 12),
                  _overviewCard(
                    icon: Icons.monitor_heart_rounded,
                    title: 'Growth',
                    subtitle: 'Health indicators',
                    color: AppColors.accent,
                  ),
                  const SizedBox(width: 12),
                  _overviewCard(
                    icon: Icons.lock_rounded,
                    title: 'Private',
                    subtitle: 'Data stays local',
                    color: AppColors.lavender,
                  ),
                ],
              ),

              const SizedBox(height: 28),

              // HISTORY + ABOUT
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _openHistory(context),
                      icon: const Icon(
                        Icons.history_rounded,
                      ),
                      label: const Text('History'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _openAbout(context),
                      icon: const Icon(
                        Icons.info_outline_rounded,
                      ),
                      label: const Text('About'),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 22),

              Center(
                child: Text(
                  'Version 1.0',
                  style: AppTextStyles.subtitle.copyWith(
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}