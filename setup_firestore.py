#!/usr/bin/env python3
"""
Setup Firestore database collections and seed initial data
"""

import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

# Load Firebase credentials
cred_path = Path('firebase-service-account.json')
if not cred_path.exists():
    print("❌ ERROR: firebase-service-account.json not found!")
    exit(1)

print(f"📋 Loading credentials from {cred_path}...")
cred = credentials.Certificate(str(cred_path))

# Initialize Firebase
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("✅ Connected to Firebase Firestore")

# Create demo patients if collection is empty
print("\n📝 Seeding demo patients...")

try:
    # Check if data exists and clear before reseeding
    existing_docs = list(db.collection('patients').stream())
    if existing_docs:
        print(f"⚠️  Found {len(existing_docs)} existing patient(s). Clearing for fresh seeding...")
        for doc in db.collection('patients').stream():
            # Delete all visits subcollections first
            for visit_doc in doc.reference.collection('visits').stream():
                visit_doc.reference.delete()
            # Delete patient document
            doc.reference.delete()
        print("✓ Cleared old data\n")
    
    demo_patients = [
            {
                'id': 'patient_001',
                'name': 'Alice Johnson',
                'dob': '1990-03-15',
                'visits': [
                    {'label': 'Visit 1 (Week 12)', 'date': (datetime.now() - timedelta(days=90)).isoformat(), 'sbp': 128, 'dbp': 82, 'bmi': 24.5, 'blood_sugar': 95, 'risk': 0.15, 'notes': 'Routine checkup'},
                    {'label': 'Visit 2 (Week 16)', 'date': (datetime.now() - timedelta(days=75)).isoformat(), 'sbp': 130, 'dbp': 84, 'bmi': 24.6, 'blood_sugar': 97, 'risk': 0.18, 'notes': 'Good progress'},
                    {'label': 'Visit 3 (Week 20)', 'date': (datetime.now() - timedelta(days=60)).isoformat(), 'sbp': 132, 'dbp': 85, 'bmi': 24.7, 'blood_sugar': 98, 'risk': 0.22, 'notes': 'Stable'},
                    {'label': 'Visit 4 (Week 24)', 'date': (datetime.now() - timedelta(days=45)).isoformat(), 'sbp': 134, 'dbp': 86, 'bmi': 24.8, 'blood_sugar': 100, 'risk': 0.25, 'notes': 'Slight increase'},
                ]
            },
            {
                'id': 'patient_002',
                'name': 'Sarah Mitchell',
                'dob': '1988-07-22',
                'visits': [
                    {'label': 'Visit 1 (Week 8)', 'date': (datetime.now() - timedelta(days=120)).isoformat(), 'sbp': 142, 'dbp': 92, 'bmi': 27.8, 'blood_sugar': 110, 'risk': 0.55, 'notes': 'Elevated baseline'},
                    {'label': 'Visit 2 (Week 12)', 'date': (datetime.now() - timedelta(days=105)).isoformat(), 'sbp': 145, 'dbp': 94, 'bmi': 27.9, 'blood_sugar': 113, 'risk': 0.62, 'notes': 'Worsening'},
                    {'label': 'Visit 3 (Week 20)', 'date': (datetime.now() - timedelta(days=80)).isoformat(), 'sbp': 148, 'dbp': 96, 'bmi': 28.1, 'blood_sugar': 117, 'risk': 0.72, 'notes': 'Close monitoring'},
                    {'label': 'Visit 4 (Week 28)', 'date': (datetime.now() - timedelta(days=55)).isoformat(), 'sbp': 150, 'dbp': 98, 'bmi': 28.3, 'blood_sugar': 120, 'risk': 0.82, 'notes': 'Critical readings'},
                    {'label': 'Visit 5 (Week 32)', 'date': (datetime.now() - timedelta(days=40)).isoformat(), 'sbp': 152, 'dbp': 100, 'bmi': 28.4, 'blood_sugar': 122, 'risk': 0.88, 'notes': 'High risk status'},
                ]
            },
            {
                'id': 'patient_003',
                'name': 'Emma Rodriguez',
                'dob': '1992-11-08',
                'visits': [
                    {'label': 'Visit 1 (Week 6)', 'date': (datetime.now() - timedelta(days=130)).isoformat(), 'sbp': 116, 'dbp': 74, 'bmi': 21.8, 'blood_sugar': 86, 'risk': 0.02, 'notes': 'Healthy baseline'},
                    {'label': 'Visit 2 (Week 14)', 'date': (datetime.now() - timedelta(days=95)).isoformat(), 'sbp': 118, 'dbp': 75, 'bmi': 22.0, 'blood_sugar': 87, 'risk': 0.04, 'notes': 'Good health'},
                    {'label': 'Visit 3 (Week 22)', 'date': (datetime.now() - timedelta(days=65)).isoformat(), 'sbp': 120, 'dbp': 77, 'bmi': 22.2, 'blood_sugar': 89, 'risk': 0.06, 'notes': 'Stable'},
                ]
            },
            {
                'id': 'patient_004',
                'name': 'Jennifer Lee',
                'dob': '1989-05-19',
                'visits': [
                    {'label': 'Visit 1 (Week 10)', 'date': (datetime.now() - timedelta(days=110)).isoformat(), 'sbp': 138, 'dbp': 88, 'bmi': 26.2, 'blood_sugar': 108, 'risk': 0.45, 'notes': 'Moderately elevated'},
                    {'label': 'Visit 2 (Week 18)', 'date': (datetime.now() - timedelta(days=80)).isoformat(), 'sbp': 140, 'dbp': 89, 'bmi': 26.4, 'blood_sugar': 110, 'risk': 0.52, 'notes': 'Stable moderate'},
                    {'label': 'Visit 3 (Week 26)', 'date': (datetime.now() - timedelta(days=50)).isoformat(), 'sbp': 143, 'dbp': 91, 'bmi': 26.6, 'blood_sugar': 112, 'risk': 0.60, 'notes': 'Monitoring needed'},
                ]
            },
            {
                'id': 'patient_005',
                'name': 'Rebecca Davis',
                'dob': '1991-09-27',
                'visits': [
                    {'label': 'Visit 1 (Week 8)', 'date': (datetime.now() - timedelta(days=115)).isoformat(), 'sbp': 125, 'dbp': 80, 'bmi': 23.5, 'blood_sugar': 92, 'risk': 0.12, 'notes': 'Borderline normal'},
                    {'label': 'Visit 2 (Week 16)', 'date': (datetime.now() - timedelta(days=85)).isoformat(), 'sbp': 128, 'dbp': 82, 'bmi': 23.6, 'blood_sugar': 94, 'risk': 0.16, 'notes': 'Slight increase'},
                    {'label': 'Visit 3 (Week 24)', 'date': (datetime.now() - timedelta(days=55)).isoformat(), 'sbp': 132, 'dbp': 84, 'bmi': 23.8, 'blood_sugar': 96, 'risk': 0.22, 'notes': 'Gradual elevation'},
                ]
            },
            {
                'id': 'patient_006',
                'name': 'Michelle Garcia',
                'dob': '1987-02-14',
                'visits': [
                    {'label': 'Visit 1 (Week 12)', 'date': (datetime.now() - timedelta(days=100)).isoformat(), 'sbp': 155, 'dbp': 102, 'bmi': 29.5, 'blood_sugar': 128, 'risk': 0.92, 'notes': 'Severe hypertension'},
                    {'label': 'Visit 2 (Week 20)', 'date': (datetime.now() - timedelta(days=70)).isoformat(), 'sbp': 158, 'dbp': 104, 'bmi': 29.7, 'blood_sugar': 130, 'risk': 0.96, 'notes': 'Critical status'},
                    {'label': 'Visit 3 (Week 28)', 'date': (datetime.now() - timedelta(days=40)).isoformat(), 'sbp': 160, 'dbp': 105, 'bmi': 29.9, 'blood_sugar': 133, 'risk': 0.98, 'notes': 'Extreme risk'},
                    {'label': 'Visit 4 (Week 36)', 'date': (datetime.now() - timedelta(days=10)).isoformat(), 'sbp': 162, 'dbp': 107, 'bmi': 30.1, 'blood_sugar': 135, 'risk': 0.99, 'notes': 'Urgent intervention'},
                ]
            },
            {
                'id': 'patient_007',
                'name': 'Laura Martinez',
                'dob': '1993-08-03',
                'visits': [
                    {'label': 'Visit 1 (Week 10)', 'date': (datetime.now() - timedelta(days=105)).isoformat(), 'sbp': 122, 'dbp': 78, 'bmi': 22.8, 'blood_sugar': 90, 'risk': 0.08, 'notes': 'Healthy start'},
                    {'label': 'Visit 2 (Week 18)', 'date': (datetime.now() - timedelta(days=75)).isoformat(), 'sbp': 124, 'dbp': 79, 'bmi': 23.0, 'blood_sugar': 91, 'risk': 0.10, 'notes': 'Stable'},
                    {'label': 'Visit 3 (Week 26)', 'date': (datetime.now() - timedelta(days=45)).isoformat(), 'sbp': 126, 'dbp': 80, 'bmi': 23.2, 'blood_sugar': 92, 'risk': 0.12, 'notes': 'Normal progression'},
                    {'label': 'Visit 4 (Week 34)', 'date': (datetime.now() - timedelta(days=15)).isoformat(), 'sbp': 128, 'dbp': 81, 'bmi': 23.4, 'blood_sugar': 93, 'risk': 0.14, 'notes': 'Excellent control'},
                ]
            },
            {
                'id': 'patient_008',
                'name': 'Patricia Thompson',
                'dob': '1986-12-10',
                'visits': [
                    {'label': 'Visit 1 (Week 9)', 'date': (datetime.now() - timedelta(days=120)).isoformat(), 'sbp': 135, 'dbp': 86, 'bmi': 25.9, 'blood_sugar': 105, 'risk': 0.38, 'notes': 'Mild elevation'},
                    {'label': 'Visit 2 (Week 17)', 'date': (datetime.now() - timedelta(days=88)).isoformat(), 'sbp': 138, 'dbp': 87, 'bmi': 26.1, 'blood_sugar': 108, 'risk': 0.42, 'notes': 'Slight increase'},
                    {'label': 'Visit 3 (Week 25)', 'date': (datetime.now() - timedelta(days=56)).isoformat(), 'sbp': 141, 'dbp': 89, 'bmi': 26.3, 'blood_sugar': 111, 'risk': 0.50, 'notes': 'Moderate elevation'},
                    {'label': 'Visit 4 (Week 33)', 'date': (datetime.now() - timedelta(days=24)).isoformat(), 'sbp': 144, 'dbp': 91, 'bmi': 26.5, 'blood_sugar': 113, 'risk': 0.58, 'notes': 'Continued rise'},
                    {'label': 'Visit 5 (Week 38)', 'date': (datetime.now() - timedelta(days=5)).isoformat(), 'sbp': 146, 'dbp': 92, 'bmi': 26.7, 'blood_sugar': 115, 'risk': 0.65, 'notes': 'High monitoring'},
                ]
            },
            {
                'id': 'patient_009',
                'name': 'Katherine White',
                'dob': '1994-06-21',
                'visits': [
                    {'label': 'Visit 1 (Week 7)', 'date': (datetime.now() - timedelta(days=125)).isoformat(), 'sbp': 119, 'dbp': 76, 'bmi': 22.4, 'blood_sugar': 88, 'risk': 0.05, 'notes': 'Excellent baseline'},
                    {'label': 'Visit 2 (Week 15)', 'date': (datetime.now() - timedelta(days=93)).isoformat(), 'sbp': 121, 'dbp': 77, 'bmi': 22.6, 'blood_sugar': 89, 'risk': 0.07, 'notes': 'Stable'},
                    {'label': 'Visit 3 (Week 23)', 'date': (datetime.now() - timedelta(days=61)).isoformat(), 'sbp': 123, 'dbp': 78, 'bmi': 22.8, 'blood_sugar': 90, 'risk': 0.09, 'notes': 'Good health'},
                ]
            },
            {
                'id': 'patient_010',
                'name': 'Victoria Brown',
                'dob': '1985-04-17',
                'visits': [
                    {'label': 'Visit 1 (Week 11)', 'date': (datetime.now() - timedelta(days=112)).isoformat(), 'sbp': 148, 'dbp': 96, 'bmi': 28.2, 'blood_sugar': 119, 'risk': 0.78, 'notes': 'High risk profile'},
                    {'label': 'Visit 2 (Week 19)', 'date': (datetime.now() - timedelta(days=82)).isoformat(), 'sbp': 150, 'dbp': 97, 'bmi': 28.4, 'blood_sugar': 121, 'risk': 0.85, 'notes': 'Critical monitoring'},
                    {'label': 'Visit 3 (Week 27)', 'date': (datetime.now() - timedelta(days=52)).isoformat(), 'sbp': 152, 'dbp': 99, 'bmi': 28.6, 'blood_sugar': 124, 'risk': 0.90, 'notes': 'Very high risk'},
                    {'label': 'Visit 4 (Week 35)', 'date': (datetime.now() - timedelta(days=22)).isoformat(), 'sbp': 154, 'dbp': 100, 'bmi': 28.8, 'blood_sugar': 126, 'risk': 0.94, 'notes': 'Extreme monitoring'},
                    {'label': 'Visit 5 (Week 39)', 'date': (datetime.now() - timedelta(days=2)).isoformat(), 'sbp': 156, 'dbp': 102, 'bmi': 29.0, 'blood_sugar': 128, 'risk': 0.96, 'notes': 'Urgent care needed'},
                ]
            },
        ]
        
        total_visits = 0
        for patient in demo_patients:
            patient_id = patient['id']
            patient_visits = patient.pop('visits', [])
            total_visits += len(patient_visits)
            
            # Create patient document
            db.collection('patients').document(patient_id).set({
                'name': patient['name'],
                'dob': patient['dob'],
                'updated_at': firestore.SERVER_TIMESTAMP
            })
            print(f"  ✓ Created patient: {patient['name']} ({patient_id})")
            
            # Create visits subcollection
            for visit in patient_visits:
                visit_label = visit['label']
                db.collection('patients').document(patient_id).collection('visits').document(visit_label).set(visit)
            print(f"    → Added {len(patient_visits)} visit(s)")
        
        print(f"\n✅ Seeded {len(demo_patients)} demo patients with {total_visits} total visits")

except Exception as e:
    print(f"❌ Error seeding patients: {e}")
    exit(1)

print("\n" + "="*60)
print("✅ FIRESTORE SETUP COMPLETE!")
print("="*60)
print("\nYour Firestore database is ready:")
print("  📦 Collection: 'patients'")
print("  └─ Subcollections: 'visits' (per patient)")
print("\nYou can now run the Streamlit app:")
print("  $ streamlit run app.py")
print("\n")
