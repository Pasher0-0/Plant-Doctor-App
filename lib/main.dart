import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'dart:io';
import 'data_model.dart'; // ดึงข้อมูลคำแนะนำมาใช้
import 'db_helper.dart';
import 'package:intl/intl.dart';

void main() => runApp(const MaterialApp(home: HomeScreen()));

// --- 1. หน้าหลัก (Home Screen) ---
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final picker = ImagePicker();
  Interpreter? _interpreter;
  List<String> _labels = [];
  int _inputHeight = 224;
  int _inputWidth = 224;
  bool _modelReady = false;

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model_unquant.tflite');
      final inputTensor = _interpreter!.getInputTensor(0);
      final shape = inputTensor.shape;
      if (shape.length >= 3) {
        _inputHeight = shape[1];
        _inputWidth = shape[2];
      }
      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData.split('\n').map((e) {
        e = e.trim();
        if (e.isEmpty) return '';
        final parts = e.split(' ');
        return parts.length > 1 ? parts.sublist(1).join(' ') : e;
      }).where((e) => e.isNotEmpty).toList();
      setState(() => _modelReady = true);
    } catch (e) {
      debugPrint('Error loading model: $e');
    }
  }

  Future<void> _pickAndAnalyzeImage() async {
    if (!_modelReady || _interpreter == null || _labels.isEmpty) return;
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      await _analyzeImage(File(pickedFile.path));
    }
  }

  /// Preprocess image to match Teachable Machine input expectations:
  /// - Center crop to square (1:1 aspect ratio)
  /// - Resize to 224x224
  /// - Normalize RGB pixels to [-1.0, 1.0]
  /// - Return processed image file and input tensor
  Future<(File processedImageFile, List<List<List<List<double>>>> inputTensor)>
      _preprocessImageForTeachableMachine(img.Image oriImage) async {
    // Step 1: Center crop to square (preserve aspect ratio as 1:1)
    final width = oriImage.width;
    final height = oriImage.height;
    final cropSize = width < height ? width : height;
    final startX = (width - cropSize) ~/ 2;
    final startY = (height - cropSize) ~/ 2;

    img.Image cropped = img.copyCrop(
      oriImage,
      x: startX,
      y: startY,
      width: cropSize,
      height: cropSize,
    );

    // Step 2: Resize to model input size (224x224) using bilinear interpolation
    // This ensures small images (e.g., 56x56) are upscaled smoothly to match Teachable Machine behavior
    img.Image resized = img.copyResize(
      cropped,
      width: _inputWidth,
      height: _inputHeight,
      interpolation: img.Interpolation.linear, // Bilinear interpolation for smooth upscaling
    );

    // Step 3: Save processed image for display
    final resizedBytes = img.encodeJpg(resized);
    final processedImageFile = File(
        '${Directory.systemTemp.path}/processed_${DateTime.now().millisecondsSinceEpoch}.jpg');
    await processedImageFile.writeAsBytes(resizedBytes);

    // Step 4: Normalize pixels to [-1.0, 1.0] and create input tensor
    // Shape: [1, 224, 224, 3] for batch=1, height=224, width=224, channels=3 (RGB)
    var inputTensor = List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (y) => List.generate(
          _inputWidth,
          (x) {
            final pixel = resized.getPixel(x, y);
            // Ensure RGB format (some pixel representations might vary)
            final r = pixel.r.toDouble();
            final g = pixel.g.toDouble();
            final b = pixel.b.toDouble();
            // Normalize: (pixel - 127.5) / 127.5 converts [0, 255] to [-1.0, 1.0]
            return [
              (r - 127.5) / 127.5,
              (g - 127.5) / 127.5,
              (b - 127.5) / 127.5,
            ];
          },
        ),
      ),
    );

    return (processedImageFile, inputTensor);
  }

  Future<void> _analyzeImage(File image) async {
    if (_interpreter == null || _labels.isEmpty) return;
    final bytes = await image.readAsBytes();
    img.Image? oriImage = img.decodeImage(bytes);
    if (oriImage == null) return;

    // Preprocess image using Teachable Machine standards
    final (processedImageFile, inputTensor) =
        await _preprocessImageForTeachableMachine(oriImage);

    var input = inputTensor;

    var output = List.filled(1, List.filled(_labels.isNotEmpty ? _labels.length : 1000, 0.0));

    try {
      _interpreter!.run(input, output);
    } catch (e) {
      debugPrint('Inference error: $e');
      return;
    }

    final probs = (output[0] as List).cast<double>();
    
    // Find top 2 predictions
    int maxIdx = 0;
    double maxVal = probs[0];
    int secondMaxIdx = -1;
    double secondMaxVal = -1;
    
    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > maxVal) {
        secondMaxIdx = maxIdx;
        secondMaxVal = maxVal;
        maxVal = probs[i];
        maxIdx = i;
      } else if (probs[i] > secondMaxVal) {
        secondMaxIdx = i;
        secondMaxVal = probs[i];
      }
    }
    
    // Use second result if top result is below threshold (75%)
    if (maxVal < 0.75 && secondMaxIdx != -1) {
      maxIdx = secondMaxIdx;
      maxVal = secondMaxVal;
    }

    String label = (_labels.isNotEmpty && maxIdx < _labels.length) ? _labels[maxIdx] : 'ไม่ทราบสาเหตุ';

    // Build full predictions list (label + probability)
    List<Map<String, dynamic>> predictions = List.generate(probs.length, (i) {
      final lab = (i < _labels.length) ? _labels[i] : 'Unknown';
      final p = probs[i];
      return {'label': lab, 'prob': p};
    });

    // Sort predictions by probability desc for display
    predictions.sort((a, b) => (b['prob'] as double).compareTo(a['prob'] as double));

    if (!mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultScreen(label: label, image: processedImageFile, predictions: predictions),
      ),
    );
  }

  @override
  void dispose() {
    try { _interpreter?.close(); } catch (_) {}
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Plant Doctor")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: double.infinity,
              height: 60,
              child: ElevatedButton.icon(
                onPressed: _modelReady ? _pickAndAnalyzeImage : null,
                icon: const Icon(Icons.camera_alt),
                label: const Text("Scan ตรวจสอบพืช", style: TextStyle(fontSize: 18)),
              ),
            ),
            const SizedBox(height: 15),
            _buildMenuButton(context, Icons.history, "ดูผลที่บันทึก", const HistoryScreen()),
            const SizedBox(height: 15),
            _buildMenuButton(context, Icons.help_outline, "คำอธิบายการใช้งาน", const InstructionScreen()),
          ],
        ),
      ),
    );
  }

  Widget _buildMenuButton(BuildContext context, IconData icon, String text, Widget page) {
    return SizedBox(
      width: double.infinity,
      height: 60,
      child: ElevatedButton.icon(
        onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (context) => page)),
        icon: Icon(icon),
        label: Text(text, style: const TextStyle(fontSize: 18)),
      ),
    );
  }
}

// --- 2. หน้าแสดงผลลัพธ์และปุ่มบันทึก ---
class ResultScreen extends StatefulWidget {
  final String label;
  final File image;
  final List<Map<String, dynamic>>? predictions;

  const ResultScreen({super.key, required this.label, required this.image, this.predictions});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _showPredictions = false;

  void _togglePredictions() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("ยืนยันการแสดงผล"),
          content: const Text("คุณต้องการดูรายละเอียดของการคาดการณ์ทั้งหมดหรือไม่?"),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("ยกเลิก"),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                setState(() => _showPredictions = true);
              },
              child: const Text("ใช่"),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    var info = PlantData.info[widget.label] ?? {"title": "ไม่ทราบสาเหตุ", "solution": "-"};

    return Scaffold(
      appBar: AppBar(title: const Text("ผลการวิเคราะห์")),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Image.file(widget.image, height: 300, width: double.infinity, fit: BoxFit.cover),
            Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("อาการ: ${info['title']}", style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.green)),
                  const SizedBox(height: 10),
                  const Text("วิธีแก้ไข/สารอาหารที่ต้องเติม:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                  Text("${info['solution']}", style: const TextStyle(fontSize: 16)),
                  const SizedBox(height: 30),

                  const SizedBox(height: 10),
                  if (widget.predictions != null) ...[
                    GestureDetector(
                      onTap: _togglePredictions,
                      child: Row(
                        children: [
                          Expanded(
                            child: const Text("ผลการคาดการณ์ทั้งหมด:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.blue)),
                          ),
                          Icon(_showPredictions ? Icons.expand_less : Icons.expand_more, color: Colors.blue),
                        ],
                      ),
                    ),
                    if (_showPredictions) ...[
                      const Text("(ถ้าค่ามากสุดต่ำกว่า 75% จะใช้ลำดับที่ 2 เป็นผลลัพธ์):", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      const SizedBox(height: 10),
                      Column(
                        children: widget.predictions!.map((p) {
                          final pLabel = p['label'] as String;
                          final pVal = (p['prob'] as double) * 100.0;
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 4.0),
                            child: Row(
                              children: [
                                Expanded(child: Text(pLabel, style: const TextStyle(fontSize: 16))),
                                Text("${pVal.toStringAsFixed(1)}%", style: const TextStyle(fontSize: 16)),
                              ],
                            ),
                          );
                        }).toList(),
                      ),
                    ],
                  ] else ...[
                    const SizedBox(height: 10),
                  ],

                  const SizedBox(height: 20),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                     onPressed: () async {
                        final now = DateTime.now();
                        final formatter = DateFormat('dd/MM/yyyy HH:mm');
                        
                        await DBHelper.insert('history', {
                          'label': widget.label,
                          'date': formatter.format(now),
                          'imagePath': widget.image.path,
                        });

                        if (!context.mounted) return;
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text("บันทึกประวัติเรียบร้อยแล้ว!")),
                        );
                        Navigator.popUntil(context, (route) => route.isFirst); // กลับหน้าแรก
                      },
                      style: ElevatedButton.styleFrom(backgroundColor: Colors.blue),
                      child: const Text("บันทึกประวัติการตรวจ", style: TextStyle(color: Colors.white)),
                    ),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}



// หน้าว่างสำหรับเมนูอื่นๆ
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("ประวัติการวิเคราะห์")),
      body: FutureBuilder(
        future: DBHelper.getData('history'),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          
          if (!snapshot.hasData || (snapshot.data as List).isEmpty) {
            return const Center(child: Text("ยังไม่มีข้อมูลการบันทึก"));
          }

          final historyList = snapshot.data as List<Map<String, dynamic>>;

          return ListView.builder(
            itemCount: historyList.length,
           itemBuilder: (context, index) {
            final item = historyList[index];
            return Dismissible(
              key: Key(item['id'].toString()),
              direction: DismissDirection.endToStart, // ปัดจากขวาไปซ้าย
              background: Container(
                color: Colors.red,
                alignment: Alignment.centerRight,
                padding: const EdgeInsets.only(right: 20),
                child: const Icon(Icons.delete, color: Colors.white),
              ),
              onDismissed: (direction) {
                DBHelper.delete(item['id']); // ลบในฐานข้อมูล
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text("ลบข้อมูลเรียบร้อยแล้ว")),
                );
              },
              child: Card(
                margin: const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
                child: ListTile(
                  leading: Image.file(
                    File(historyList[index]['imagePath']),
                    width: 50, height: 50, fit: BoxFit.cover,
                  ),
                  title: Text(historyList[index]['label']),
                  subtitle: Text(historyList[index]['date']),
                  onTap: () {
                    final imagePath = historyList[index]['imagePath'] as String;
                    final label = historyList[index]['label'] as String;
                    final imageFile = File(imagePath);
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ResultScreen(label: label, image: imageFile),
                      ),
                    );
                  },
                ),
              ),
            );

            },
          );
        },
      ),
    );
  }
}

class InstructionScreen extends StatelessWidget {
  const InstructionScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("วิธีใช้งาน")),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _buildStep(Icons.crop, "Step 0", "ครอบรูปภาพที่ถ่ายให้เห็นส่วนใบข้าวที่มีอาการผิดปกติให้ชัดเจน"),
          _buildStep(Icons.photo_library, "Step 1", "กดปุ่ม Scan เพื่อเข้าสู่หน้าวิเคราะห์"),
          _buildStep(Icons.folder_open, "Step 2", "เลือกรูปภาพใบข้าวจากแกลเลอรี่"),
          _buildStep(Icons.image_search, "Step 3", "ระบบจะประมวลผลรูปภาพ (ปรับขนาด ปรับสี) เพื่อให้เหมาะสมกับโมเดล"),
          _buildStep(Icons.analytics, "Step 4", "ระบบ AI จะวิเคราะห์และหาสาเหตุที่เป็นไปได้ของอาการใบข้าว"),
          _buildStep(Icons.help, "Step 5", "อ่านผลลัพธ์และคำแนะนำในการปรับปรุงพืช"),
          const SizedBox(height: 24),
          Center(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final imgHeight = MediaQuery.of(context).size.height * 0.2;
                return Image.asset(
                  'assets/test_data/P (3).jpg', // เปลี่ยน path ตรงนี้ถ้ารูปอยู่ที่อื่น
                  height: imgHeight,
                  fit: BoxFit.contain,
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStep(IconData icon, String title, String desc) {
    return ListTile(
      leading: CircleAvatar(child: Icon(icon)),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
      subtitle: Text(desc),
    );
  }
}