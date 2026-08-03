# Court Area Definitions
The half court is divided into 17 semantic areas.
## Deep Areas (Out of the Baseline) 
- **Deep Left Corner**: 
  - Semantics: Left corner area out of the baseline
  - Condition: 50 <= x <= 177 AND 0 <= y <= 150
- **Deep Right Corner**: 
  - Semantics: Right corner area out of the baseline
  - Condition: 178 <= x <= 305 AND 0 <= y <= 150
## Rear Areas (Back Court)
- **Rear Left Side**:
  - Semantics: Left-hand side of the inner rear court.
  - Condition: 50 <= x <= 103 AND 151 <= y <= 257
- **Rear Center**: 
  - Semantics: Central zone of the rear court around the center line.
  - Condition: 104 <= x <= 250 AND 151 <= y <= 257
- **Rear Right Side**: 
  - Semantics: Right-hand side of the inner rear court.
  - Condition: 251 <= x <= 304 AND 151 <= y <= 257
- **Rear Left Side Line**: 
  - Semantics: Outer left tramline/sideline area of the back court.
  - Condition: 0 <= x <= 49 AND 0 <= y <= 204
- **Rear Right Side Line**: 
  - Semantics: Outer right tramline/sideline area of the back court.
  - Condition: 305 <= x <= 355 AND 0 <= y <= 204
## Mid Areas (Middle Court)
- **Mid Left Side**:  
  - Semantics: Left-hand side of the inner middle court.
  - Condition: 50 <= x <= 103 AND 258 <= y <= 365
- **Mid Center**:  
  - Semantics: Central zone of the middle court.
  - Condition: 104 <= x <= 250 AND 258 <= y <= 365
- **Mid Right Side**:  
  - Semantics: Right-hand side of the inner middle court.
  - Condition: 251 <= x <= 304 AND 258 <= y <= 365
- **Mid Left Side Line**:  
  - Semantics: Outer left tramline/sideline area of the middle court.
  - Condition: 0 <= x <= 49 AND 205 <= y <= 365
- **Mid Right Side Line**:  
  - Semantics: Outer right tramline/sideline area of the middle court.
  - Condition: 305 <= x <= 355 AND 205 <= y <= 365
## Front Areas (Front Court)
- **Front Left Side**: 
  - Semantics: Left-hand side of the inner front court near the net.
  - Condition: 50 <= x <= 103 AND 366 <= y <= 480
- **Front Right Side**:  
  - Semantics: Right-hand side of the inner front court near the net.
  - Condition: 251 <= x <= 304 AND 366 <= y <= 480
- **Front Center**:  
  - Semantics: Central net area / short service zone.
  - Condition: 104 <= x <= 250 AND 366 <= y <= 480
- **Front Left Side Line**:  
  - Semantics: Outer left tramline/sideline area of the front court.
  - Condition: 0 <= x <= 49 AND 366 <= y <= 480
- **Front Right Side Line**:  
  - Semantics: Outer right tramline/sideline area of the front court.
  - Condition: 305 <= x <= 355 AND 366 <= y <= 480
# Coordinate System
- x increases from left to right.
- y increases from the baseline toward the net.
- Valid grid coordinates satisfy 0 <= x <= 355 and 0 <= y <= 480.
- All non-missed landing and recovery coordinates must be integers.
- The semantic area must match its integer coordinate.

# Shot Types
## Serve
- **Serve short**
- **Serve long**
## Attack
- **Push Shot**
- **Smash**
- **Drop**
## Defense
- **Clear**
- **Smash Defence**
- **Lob**
## Neutral
- **Drive**
- **Net Shot**