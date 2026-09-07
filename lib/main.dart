import 'package:flutter/material.dart';
import 'app.dart';
import 'services/onnx_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final onnxService = OnnxService();

  await onnxService.initialize();

  runApp(const MalnutritionApp());
}