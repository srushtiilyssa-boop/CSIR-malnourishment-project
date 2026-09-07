import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../models/patient.dart';

class HistoryService {
  static const String fileName = 'screening_history.json';

  static Future<File> _getFile() async {
    final directory = await getApplicationDocumentsDirectory();

    final file = File(
      '${directory.path}/$fileName',
    );

    if (!await file.exists()) {
      await file.writeAsString('[]');
    }

    return file;
  }

  static Future<List<Map<String, dynamic>>> getHistory() async {
    try {
      final file = await _getFile();

      final content = await file.readAsString();

      if (content.trim().isEmpty) {
        return [];
      }

      final decoded = jsonDecode(content);

      if (decoded is! List) {
        return [];
      }

      return decoded
          .whereType<Map>()
          .map(
            (item) => Map<String, dynamic>.from(item),
          )
          .toList();
    } catch (e) {
      print('History read error: $e');
      return [];
    }
  }

  static Future<void> saveScreening({
    required Patient patient,
    required String prediction,
    required double confidence,
    required int classIndex,
    required List<double> probabilities,
  }) async {
    try {
      final file = await _getFile();

      final history = await getHistory();

      final record = <String, dynamic>{
        'date': DateTime.now().toIso8601String(),

        'ageMonths': patient.childAgeMonths,

        'gender': patient.gender,

        'weightKg': patient.weightKg,

        'heightCm': patient.heightCm,

        'prediction': prediction,

        'confidence': confidence,

        'classIndex': classIndex,

        'probabilities': probabilities,
      };

      history.insert(0, record);

      await file.writeAsString(
        jsonEncode(history),
      );

      print('Screening saved to history.');
    } catch (e) {
      print('History save error: $e');
    }
  }

  static Future<void> clearHistory() async {
    try {
      final file = await _getFile();

      await file.writeAsString('[]');

      print('Screening history cleared.');
    } catch (e) {
      print('History clear error: $e');
    }
  }
}