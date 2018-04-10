import zipfile
import sqlite3
import unicodedata

zipfile.ZipFile("database.sqlite.zip", 'r').extractall()
zip_ref.close()

conn = sqlite3.connect("database.sqlite")
conn.text_factory = str
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
cur.execute("PRAGMA table_info(content)").fetchall()

cur.execute("SELECT content FROM content")
pitchfork = cur.fetchall()
print(pitchfork[0])
conn.close()


def normalize_unicode(query):
    data = []
    for row in query:
        data.append(unicodedata.normalize("NFKD", row[0]))
    return data

pitchfork = normalize_unicode(pitchfork)


file = open('pitchfork.txt', 'w')
for review in pitchfork:
  file.write("%s\n" % review)
