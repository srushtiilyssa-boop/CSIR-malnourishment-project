import 'package:flutter/material.dart';

import '../theme/app_colors.dart';
import '../theme/app_text_styles.dart';
import '../services/history_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  bool loading = true;

  List<Map<String, dynamic>> records = [];

  @override
  void initState() {
    super.initState();
    loadHistory();
  }

  Future<void> loadHistory() async {
    final data = await HistoryService.getHistory();

    if (!mounted) return;

    setState(() {
      records = data;
      loading = false;
    });
  }

  Future<void> clearHistory() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Clear History?'),
          content: const Text(
            'This will remove all saved screening records from this device.',
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context, false);
              },
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () {
                Navigator.pop(context, true);
              },
              child: const Text('Clear'),
            ),
          ],
        );
      },
    );

    if (confirm != true) return;

    await HistoryService.clearHistory();

    await loadHistory();
  }

  String formatDate(dynamic value) {
    try {
      final date = DateTime.parse(
        value.toString(),
      ).toLocal();

      final day =
          date.day.toString().padLeft(2, '0');

      final month =
          date.month.toString().padLeft(2, '0');

      final hour =
          date.hour.toString().padLeft(2, '0');

      final minute =
          date.minute.toString().padLeft(2, '0');

      return '$day/$month/${date.year}  $hour:$minute';
    } catch (_) {
      return 'Unknown date';
    }
  }

  Color resultColor(String prediction) {
    final text = prediction.toLowerCase();

    if (text.contains('high') ||
        text.contains('malnourished') ||
        text.contains('severe')) {
      return Colors.red;
    }

    if (text.contains('moderate')) {
      return Colors.orange;
    }

    if (text.contains('low') ||
        text.contains('nourished')) {
      return Colors.green;
    }

    return AppColors.primary;
  }

  IconData resultIcon(String prediction) {
    final text = prediction.toLowerCase();

    if (text.contains('high') ||
        text.contains('malnourished') ||
        text.contains('severe')) {
      return Icons.warning_rounded;
    }

    if (text.contains('moderate')) {
      return Icons.info_rounded;
    }

    return Icons.check_circle_rounded;
  }

  Widget detail(
    String label,
    String value,
  ) {
    return Expanded(
      child: Column(
        children: [
          Text(
            label,
            style: AppTextStyles.subtitle.copyWith(
              fontSize: 11,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            textAlign: TextAlign.center,
            style: AppTextStyles.title.copyWith(
              fontSize: 13,
            ),
          ),
        ],
      ),
    );
  }

  Widget historyCard(
    Map<String, dynamic> record,
  ) {
    final prediction =
        record['prediction']?.toString() ??
            'Unknown';

    final confidence =
        (record['confidence'] as num?)
                ?.toDouble() ??
            0.0;

    final age =
        record['ageMonths']?.toString() ??
            '-';

    final gender =
        record['gender']?.toString() ??
            '-';

    final weight =
        (record['weightKg'] as num?)
            ?.toDouble();

    final height =
        (record['heightCm'] as num?)
            ?.toDouble();

    final color = resultColor(prediction);

    return Container(
      padding: const EdgeInsets.all(18),
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
        children: [
          Row(
            children: [
              Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.12),
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  resultIcon(prediction),
                  color: color,
                  size: 28,
                ),
              ),

              const SizedBox(width: 14),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      prediction,
                      style: AppTextStyles.title.copyWith(
                        color: color,
                        fontSize: 18,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      formatDate(record['date']),
                      style:
                          AppTextStyles.subtitle.copyWith(
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),

              Text(
                '${(confidence * 100).toStringAsFixed(1)}%',
                style: AppTextStyles.title.copyWith(
                  fontSize: 17,
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),

          const Divider(),

          const SizedBox(height: 12),

          Row(
            children: [
              detail(
                'Age',
                '$age months',
              ),
              detail(
                'Gender',
                gender,
              ),
              detail(
                'Weight',
                weight == null
                    ? '-'
                    : '${weight.toStringAsFixed(1)} kg',
              ),
              detail(
                'Height',
                height == null
                    ? '-'
                    : '${height.toStringAsFixed(1)} cm',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget emptyHistory() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(30),
        child: Column(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children: [
            Container(
              width: 85,
              height: 85,
              decoration: BoxDecoration(
                color:
                    AppColors.primary.withOpacity(0.12),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.history_rounded,
                size: 42,
                color: AppColors.primary,
              ),
            ),

            const SizedBox(height: 18),

            Text(
              'No screenings yet',
              style: AppTextStyles.title.copyWith(
                fontSize: 20,
              ),
            ),

            const SizedBox(height: 7),

            Text(
              'Completed screening results will appear here automatically.',
              textAlign: TextAlign.center,
              style: AppTextStyles.subtitle,
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,

      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,

        title: Text(
          'Screening History',
          style: AppTextStyles.title,
        ),

        actions: [
          if (records.isNotEmpty)
            IconButton(
              tooltip: 'Clear history',
              onPressed: clearHistory,
              icon: const Icon(
                Icons.delete_outline_rounded,
              ),
            ),
        ],
      ),

      body: loading
          ? const Center(
              child: CircularProgressIndicator(),
            )
          : records.isEmpty
              ? emptyHistory()
              : RefreshIndicator(
                  onRefresh: loadHistory,
                  child: ListView.separated(
                    padding: const EdgeInsets.all(18),
                    itemCount: records.length,
                    separatorBuilder: (_, __) =>
                        const SizedBox(height: 12),
                    itemBuilder: (context, index) {
                      return historyCard(
                        records[index],
                      );
                    },
                  ),
                ),
    );
  }
}