# Replacement Types

## Replace with a Single Word/Token

- **ColorLang:** "color", "shade", "hue"
- **TempLang:** "time", "moment", "instant"
- **KinLang:** "relative", "kin"
- **NumLang:** "number", "quant", "metric", "value"
- **LogicLang:** "logic", "link"
- **ShapeLang:** "shape", "form"
- **SpaceLang:** "direction", "place"
- **GenderLang:** "person"
- **EmotionLang:** "emotion", "feeling"
- **ValenceLang:** "valence"

## Replace with a Tag

- **ColorLang:** <"color">
- **TempLang:** <"time">
- **KinLang:** <"relative">
- **NumLang:** <"number">
- **LogicLang:** <"logic">
- **ShapeLang:** <"shape">
- **SpaceLang:** <"direction">
- **GenderLang:** <"person">
- **EmotionLang:** <"emotion">
- **ValenceLang:** <"valence">

## Replace with Categorical Words

- **SpaceLang:** "north", "south", "east", "west"
- **TempLang:** "day", "month", "season"
- **KinLang:** "relative1", "relative2", "relative3"
- **NumLang:** "number", "spelled_number", "ordinal", "spelled_ordinal"
- **LogicLang:** "connective", "conditional", "causal"
- **ColorLang:** "color1", "color2", "color3", "color4"
- **ShapeLang:** "shape1", "shape2", "shape3", "shape4"
- **GenderLang:** "type1", "type2"
- **EmotionLang:** "positive", "negative", "neutral"
- **ValenceLang:** "positive", "negative", "neutral"

# 📊 Replacement Statistics Across Artificial Languages

## SpaceLang  
- **Train:** 939,636 replacements  
- **Dev:** 101,554 replacements  

## TempLang  
- **Train:** 615,341 replacements  
- **Dev:** 65,300 replacements  

## KinLang  
- **Train:** 364,646 replacements  
- **Dev:** 35,911 replacements  

## NumLang  
- **Train:** 1,994,799 replacements  
- **Dev:** 197,836 replacements  

## LogicLang  
- **Train:** 3,696,195 replacements  
- **Dev:** 404,957 replacements  

## ColorLang  
- **Train:** 232,762 replacements  
- **Dev:** 20,655 replacements  

## ShapeLang  
- **Train:** 29,012 replacements  
- **Dev:** 2,691 replacements  

## GenderLang  
- **Train:** 2,638,246 replacements  
- **Dev:** 259,238 replacements  

## EmotionLang  
- **Train:** 167,111 replacements  
- **Dev:** 16,588 replacements  

## ValenceLang  
- **Train:** 15,431,146 replacements  
- **Dev:** 1,659,425 replacements  

---

# ⚠️ Limitations
- No coreference resolution used (would require ~380 GPU hours).
- Number of replacements varies across artificial languages.
