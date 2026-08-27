INSERT INTO roles (name, access_levels) VALUES
    ('clinician', ARRAY['standard']),
    ('compliance_admin', ARRAY['standard', 'restricted']);

INSERT INTO users (username, role_id) VALUES
    ('alice_clinician', (SELECT id FROM roles WHERE name = 'clinician')),
    ('bob_admin', (SELECT id FROM roles WHERE name = 'compliance_admin'));

INSERT INTO documents (doc_id, title, category, access_level) VALUES
    ('ncd_57_bariatric_surgery_for_treatment_of_co_morbid_condi.txt', 'NCD 57 - Bariatric Surgery for Treatment of Co-Morbid Conditions Related to Morbid Obesity', 'bariatric', 'standard'),
    ('lcd_35022_bariatric_surgical_management_of_morbid_obesity.txt', 'LCD 35022 - Bariatric Surgical Management of Morbid Obesity', 'bariatric', 'standard'),
    ('ncd_169_home_use_of_oxygen.txt', 'NCD 169 - Home Use of Oxygen', 'oxygen', 'standard'),
    ('lcd_33797_oxygen_and_oxygen_equipment.txt', 'LCD 33797 - Oxygen and Oxygen Equipment', 'oxygen', 'restricted'),
    ('lcd_36575_total_knee_arthroplasty.txt', 'LCD 36575 - Total Knee Arthroplasty', 'knee', 'standard'),
    ('bpm_ch1_sec10.2_admission_order_and_certification.txt', 'Benefit Policy Manual Ch. 1 Sec. 10.2 - Hospital Inpatient Admission Order and Certification', 'general_policy', 'standard'),
    ('bpm_ch15_sec110_durable_medical_equipment_general.txt', 'Benefit Policy Manual Ch. 15 Sec. 110 - Durable Medical Equipment, General', 'general_policy', 'standard'),
    ('bariatric_surgery.txt', 'Sample Coverage Policy (synthetic) - Bariatric Surgery', 'bariatric', 'standard'),
    ('home_oxygen_therapy.txt', 'Sample Coverage Policy (synthetic) - Home Oxygen Therapy', 'oxygen', 'standard'),
    ('knee_replacement.txt', 'Sample Coverage Policy (synthetic) - Total Knee Arthroplasty', 'knee', 'standard');
