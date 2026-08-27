"""Crew-facing planting briefs in Indian languages.

The full technical brief (planting_brief.txt) is written for planners, in
English. The people who actually put trees in the ground read Hindi or
Gujarati — so alongside it, the engine writes a one-page summary per
configured language: for each priority zone, where it is, what is happening
to its air and green cover, which species to plant, and the one caveat that
must survive translation (confirm every spot on the ground before digging).

Deliberately TEMPLATED, not machine-translated: every sentence below was
written by a person, once, and only numbers are substituted — so the Hindi
and Gujarati text is as trustworthy as the English and works fully offline.
The LLM's free-text English justification is not machine-translated into
these files precisely because a wrong translation on a work order is worse
than none.

Configured via `run.brief_languages` (default: hi, gu). Adding a language =
adding one dict here.
"""

from __future__ import annotations

from typing import Any

# Direction words keyed by "is the metric getting worse?"
_T: dict[str, dict[str, str]] = {
    "hi": {
        "title": "वृक्षारोपण संक्षेप — {city} (शीर्ष {n} क्षेत्र)",
        "note": (
            "पूरा तकनीकी विवरण अंग्रेज़ी में planting_brief.txt में है।\n"
            "हर स्थल की खुदाई से पहले ज़मीन पर जाकर पुष्टि करें — स्वामित्व,\n"
            "पाइपलाइन/केबल और रास्ते का मॉडल नहीं बनाया गया है।"
        ),
        "zone": (
            "प्राथमिकता {rank}: क्षेत्र {zone}\n"
            "    स्थान: {lat:.4f}, {lon:.4f} के आसपास (~5 वर्ग किमी)\n"
            "    हवा: AQI {aqi:.0f}, अगले वर्ष {aqi_delta:+.0f} अनुमानित ({aqi_dir})\n"
            "    हरियाली: {ndvi_dir} (प्रति वर्ष {ndvi_yr:+.3f})\n"
            "    लगाएँ: {species}\n"
            "    चिन्हित खाली स्थल: {n_sites}"
        ),
        "worsening": "बिगड़ रही है",
        "improving": "सुधर रही है",
        "declining": "घट रही है",
        "growing": "बढ़ रही है",
        "no_species": "प्रजातियाँ अंग्रेज़ी विवरण में देखें",
    },
    "gu": {
        "title": "વૃક્ષારોપણ સાર — {city} (ટોચના {n} વિસ્તાર)",
        "note": (
            "સંપૂર્ણ તકનીકી વિગત અંગ્રેજીમાં planting_brief.txt માં છે.\n"
            "દરેક સ્થળે ખોદકામ પહેલાં જમીન પર જઈ ખાતરી કરો — માલિકી,\n"
            "પાઇપલાઇન/કેબલ અને રસ્તાનું મૉડેલ બનાવેલું નથી."
        ),
        "zone": (
            "અગ્રતા {rank}: વિસ્તાર {zone}\n"
            "    સ્થળ: {lat:.4f}, {lon:.4f} આસપાસ (~5 ચો. કિમી)\n"
            "    હવા: AQI {aqi:.0f}, આવતા વર્ષે {aqi_delta:+.0f} અંદાજ ({aqi_dir})\n"
            "    હરિયાળી: {ndvi_dir} (દર વર્ષે {ndvi_yr:+.3f})\n"
            "    વાવો: {species}\n"
            "    ચિહ્નિત ખાલી સ્થળ: {n_sites}"
        ),
        "worsening": "બગડી રહી છે",
        "improving": "સુધરી રહી છે",
        "declining": "ઘટી રહી છે",
        "growing": "વધી રહી છે",
        "no_species": "પ્રજાતિઓ અંગ્રેજી વિગતમાં જુઓ",
    },
}

SUPPORTED = sorted(_T)


def crew_brief(
    lang: str,
    city: str,
    zones: list[dict[str, Any]],
) -> str:
    """One-page crew summary. Each zone dict needs: rank, zone, lat, lon,
    aqi_latest, aqi_pred_delta, ndvi_slope, species (list), n_sites (int)."""
    t = _T[lang]
    lines = [t["title"].format(city=city, n=len(zones)), "=" * 44, ""]
    for z in zones:
        species = ", ".join(z["species"]) if z.get("species") else t["no_species"]
        lines.append(
            t["zone"].format(
                rank=z["rank"],
                zone=z["zone"],
                lat=z["lat"],
                lon=z["lon"],
                aqi=z["aqi_latest"],
                aqi_delta=z["aqi_pred_delta"],
                aqi_dir=t["worsening"] if z["aqi_pred_delta"] >= 0 else t["improving"],
                ndvi_dir=t["declining"] if z["ndvi_slope"] < 0 else t["growing"],
                ndvi_yr=z["ndvi_slope"] * 12,
                species=species,
                n_sites=z.get("n_sites", 0),
            )
        )
        lines.append("")
    lines += [t["note"], ""]
    return "\n".join(lines)
