#!/usr/bin/env python3
"""Generate 2500 synthetic patients with visit data and export to CSV"""
import csv
import random
from datetime import datetime, timedelta

# Lists for generating realistic names
first_names = [
    "Alice", "Sarah", "Emma", "Jennifer", "Rebecca", "Michelle", "Laura", "Patricia", "Katherine", "Victoria",
    "Amara", "Zainab", "Fatimata", "Nia", "Adanna", "Kwasiba", "Thandiwe", "Hadiya", "Zuri", "Ama",
    "Maria", "Anna", "Elena", "Diana", "Isabel", "Rosa", "Julia", "Sofia", "Lucia", "Carmen",
    "Linda", "Barbara", "Nancy", "Margaret", "Susan", "Dorothy", "Helen", "Diane", "Joyce", "Victoria"
]

last_names = [
    "Johnson", "Mitchell", "Rodriguez", "Lee", "Davis", "Garcia", "Martinez", "Thompson", "White", "Brown",
    "Okafor", "Abdi", "Diallo", "Mwangi", "Umoh", "Mensah", "Mthembu", "Mohamed", "Kangwena", "Asante",
    "Garcia", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Sanchez", "Perez", "Torres", "Rivera", "Mendoza",
    "Smith", "Williams", "Jones", "Taylor", "Anderson", "Thomas", "Moore", "Jackson", "Martin", "Lee"
]

risk_ranges = [
    (0.02, 0.06),   # Very low
    (0.08, 0.14),   # Low
    (0.15, 0.25),   # Low-moderate
    (0.38, 0.50),   # Moderate
    (0.52, 0.65),   # Moderate-high
    (0.68, 0.82),   # High
    (0.85, 0.94),   # Very high
    (0.92, 0.99),   # Critical
]

def generate_random_dob():
    """Generate random date of birth for ages 25-55"""
    today = datetime.now()
    age_range = random.randint(25, 55)
    years_ago = age_range
    months_ago = random.randint(0, 11)
    days_ago = random.randint(0, 28)
    
    dob = today - timedelta(days=years_ago*365 + months_ago*30 + days_ago)
    return dob.strftime("%Y-%m-%d")

def generate_patients_csv(filename, num_patients=2500, visits_per_patient_range=(3, 5)):
    """Generate synthetic patient data and save to CSV"""
    
    print(f"Generating {num_patients} synthetic patients...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'patient_id', 'name', 'dob', 'visit_number', 'visit_label', 
            'date', 'sbp', 'dbp', 'bmi', 'blood_sugar', 'risk', 'notes'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_visits = 0
        
        for p in range(1, num_patients + 1):
            patient_id = f"patient_{p:05d}"
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            name = f"{first_name} {last_name}"
            dob = generate_random_dob()
            
            # Random number of visits per patient
            num_visits = random.randint(visits_per_patient_range[0], visits_per_patient_range[1])
            
            for v in range(num_visits):
                week = 6 + (v * 8)
                days_ago = 130 - (v * 30)
                risk_range = risk_ranges[v % len(risk_ranges)]
                risk = risk_range[0] + (random.random() * (risk_range[1] - risk_range[0]))
                
                visit_data = {
                    'patient_id': patient_id,
                    'name': name,
                    'dob': dob,
                    'visit_number': v + 1,
                    'visit_label': f'Visit {v+1} (Week {week})',
                    'date': (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S"),
                    'sbp': 115 + (v * 4) + int(random.random() * 10),
                    'dbp': 74 + (v * 3) + int(random.random() * 8),
                    'bmi': round(22 + (v * 0.3) + (random.random() * 1), 2),
                    'blood_sugar': 85 + (v * 5) + int(random.random() * 8),
                    'risk': round(risk, 2),
                    'notes': f'Follow-up visit {v+1}'
                }
                
                writer.writerow(visit_data)
                total_visits += 1
            
            if p % 250 == 0:
                print(f"  ✓ Generated {p} patients ({total_visits} visits so far)...")
    
    print(f"\n✅ DONE: {num_patients} patients with {total_visits} total visits")
    print(f"📊 CSV file saved: {filename}")
    print(f"📈 Average visits per patient: {total_visits / num_patients:.2f}")

if __name__ == "__main__":
    generate_patients_csv("synthetic_patients_2500.csv", num_patients=2500)
