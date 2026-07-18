# Court Area Definitions

The badminton court is divided into 18 semantic areas:

## Deep Areas (Out of the Baseline) 
- **Deep Left Corner**: 
  - Semantics: Left corner area at the baseline.
  - Condition: -0.718 < x <= 0.0 AND -1.0 < y <= -0.375
- **Deep Right Corner**:
  - Semantics: Right corner area at the baseline.
  - Condition: 0.0 <= x < 0.718 AND -1.0 < y <= -0.375

# Rear Areas (Back Court)
- **Rear Left Side**:
  - Semantics: Left-hand side of the inner rear court.
  - Condition: -0.718 < x <= -0.414 AND -0.375 < y <= -0.075
- **Rear Right Side**: 
  - Semantics: Right-hand side of the inner rear court.
  - Condition: 0.414 <= x < 0.718 AND -0.375 < y <= -0.075
- **Rear Center**: 
  - Semantics: Central zone of the rear court around the center line.
  - Condition: -0.414 < x <= 0.414 AND -0.375 < y <= -0.075
- **Rear Left Side Line**: 
  - Semantics: Outer left tramline/sideline area of the back court.
  - Condition: -1.0 < x <= -0.718 AND -1.0 < y <= -0.375
- **Rear Right Side Line**: 
  - Semantics: Outer right tramline/sideline area of the back court.
  - Condition: 0.718 <= x < 1.0 AND -1.0 < y <= -0.375

## Mid Areas (Middle Court)
- **Mid Left Side**:  
  - Semantics: Left-hand side of the inner middle court.
  - Condition: -0.718 < x <= -0.414 AND -0.075 <= y < 0.525
- **Mid Right Side**:  
  - Semantics: Right-hand side of the inner middle court.
  - Condition: 0.414 <= x < 0.718 AND -0.075 <= y < 0.525
- **Mid Center**:  
  - Semantics: Central zone of the middle court.
  - Condition: -0.414 < x <= 0.414 AND -0.075 <= y < 0.525
- **Mid Left Side Line**:  
  - Semantics: Outer left tramline/sideline area of the middle court.
  - Condition: -1.0 < x <= -0.718 AND -0.15 < y < 0.525
- **Mid Right Side Line**:  
  - Semantics: Outer right tramline/sideline area of the middle court.
  - Condition: 0.718 <= x < 1.0 AND -0.15 < y < 0.525

## Front Areas (Front Court)
- **Front Left Side**: 
  - Semantics: Left-hand side of the inner front court near the net.
  - Condition: -0.718 < x <= -0.414 AND 0.525 <= y < 1.0
- **Front Right Side**:  
  - Semantics: Right-hand side of the inner front court near the net.
  - Condition: 0.414 <= x < 0.718 AND 0.525 <= y < 1.0
- **Front Center**:  
  - Semantics: Central net area / short service zone.
  - Condition: -0.414 < x <= 0.414 AND 0.525 <= y < 1.0
- **Front Left Side Line**:  
  - Semantics: Outer left tramline/sideline area of the front court.
  - Condition: -1.0 < x <= -0.718 AND 0.525 <= y < 1.0
- **Front Right Side Line**:  
  - Semantics: Outer right tramline/sideline area of the front court.
  - Condition: 0.718 <= x < 1.0 AND 0.525 <= y < 1.0

## Special Areas
- **Out of Bounds**: 
  - Semantics: Any position outside or EXACTLY ON the outer boundary lines of the court.
  - Condition: x >= 1.0 OR x <= -1.0 OR y >= 1.0 OR y <= -1.0

## Coordinate System
- **x**: -1.0 (left) to 1.0 (right)
- **y**: -1.0 (baseline) to 1.0 (net)
- Normalized: -1 to 1 for both x and y

# Shot Types
- **Serve short**
- **Serve long**
- **Clear**
- **Push Shot**
- **Smash**
- **Smash Defence**
- **Drive**
- **Net Shot**
- **Lob**
- **Drop**
- **Missed shot**
