import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DBHelper {
  static Future<Database> database() async {
    final dbPath = await getDatabasesPath();
    return openDatabase(
      join(dbPath, 'plant_history.db'),
      onCreate: (db, version) {
        // สร้างตารางเก็บข้อมูล: id, ชื่ออาการ, วันที่, และที่เก็บรูป
        return db.execute(
          'CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, date TEXT, imagePath TEXT)',
        );
      },
      version: 1,
    );
  }

  // ฟังก์ชันบันทึกข้อมูล
  static Future<void> insert(String table, Map<String, dynamic> data) async {
    final db = await DBHelper.database();
    await db.insert(table, data, conflictAlgorithm: ConflictAlgorithm.replace);
  }

  // ฟังก์ชันดึงข้อมูลทั้งหมด
  static Future<List<Map<String, dynamic>>> getData(String table) async {
    final db = await DBHelper.database();
    return db.query(table, orderBy: "id DESC"); // เอาอันใหม่ขึ้นก่อน
  }

  static Future<void> delete(int id) async {
  final db = await DBHelper.database();
  await db.delete('history', where: 'id = ?', whereArgs: [id]);
}
}