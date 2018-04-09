import zipfile
import sqlite3


zipfile.ZipFile("database.sqlite.zip", 'r').extractall()
zip_ref.close()

conn = sqlite3.connect("database.sqlite")
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
cur.execute("PRAGMA table_info(content)").fetchall()

cur.execute("SELECT content FROM content")
pitchfork = cur.fetchall()
print(pitchfork[0])
conn.close()

file = open('pitchfork.txt', 'w')
for review in pitchfork:
  file.write("%s\n" % review)