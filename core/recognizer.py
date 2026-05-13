"""
core/recognizer.py
MongoDB-backed face recognition database.

Schema
------
Database: cctv_pro
Collection: persons
Document structure:
{
  "name": "Jane",
  "encoding": [0.1, 0.2, ...],
  "image_path": "uploads/database/Jane/photo.jpg"
}
"""

import os
from typing import Optional

import numpy as np
import face_recognition
from pymongo import MongoClient

# ---------------------------------------------------------------------------
# MongoDB Connection Config
# ---------------------------------------------------------------------------
# IMPORTANT: If you used MongoDB Atlas (Cloud), REPLACE the string below 
# with your actual connection string!
# For example: MONGO_URI = "mongodb+srv://admin:myPassword123@cluster0.abcde.mongodb.net/"
# 
# If you installed MongoDB locally on your PC, leave it as is.
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017/"

client = MongoClient(MONGO_URI)
db = client["cctvpro"]
persons_collection = db["persons"]


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Initialize the MongoDB index if necessary."""
    # Collections are created automatically in MongoDB upon first insertion,
    # but we can ensure an index exists on 'name' for faster lookups.
    persons_collection.create_index("name")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def add_person(name: str, encoding: np.ndarray,
               image_path: str = None) -> None:
    """Insert a face encoding for *name* into the database."""
    persons_collection.insert_one({
        "name": name,
        "encoding": encoding.tolist(),
        "image_path": image_path
    })


def delete_person(name: str) -> None:
    """Remove all documents for *name* from the database."""
    persons_collection.delete_many({"name": name})


def get_all_people() -> list:
    """
    Return a list of unique persons with their first stored image.
    [{"name": str, "image_path": str | None, "photo_count": int}, ...]
    """
    pipeline = [
        {
            "$group": {
                "_id": "$name",
                "thumbnail": {"$first": "$image_path"},
                "photo_count": {"$sum": 1}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]
    
    results = []
    for doc in persons_collection.aggregate(pipeline):
        results.append({
            "name": doc["_id"],
            "image_path": doc.get("thumbnail"),
            "photo_count": doc.get("photo_count", 0)
        })
    return results


def get_person_photos(name: str) -> list:
    """Return all image paths enrolled for *name*."""
    cursor = persons_collection.find({"name": name, "image_path": {"$ne": None}})
    return [doc["image_path"] for doc in cursor]


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def load_all_encodings() -> tuple:
    """
    Load every encoding from the database.
    Returns (encodings: list[np.ndarray], names: list[str]).
    """
    cursor = persons_collection.find({}, {"name": 1, "encoding": 1, "_id": 0})
    encodings = []
    names = []
    
    for doc in cursor:
        names.append(doc["name"])
        encodings.append(np.array(doc["encoding"]))
        
    return encodings, names


def recognize(encoding: np.ndarray,
              tolerance: float = 0.50) -> Optional[str]:
    """
    Compare *encoding* against the database.

    Returns the matched person's name, or None if no match found.
    """
    encodings, names = load_all_encodings()
    if not encodings:
        return None

    distances = face_recognition.face_distance(encodings, encoding)
    best_idx = int(np.argmin(distances))

    if distances[best_idx] <= tolerance:
        return names[best_idx]
    return None
