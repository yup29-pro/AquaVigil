"""
backend/app/knowledge.py

Aquaculture Knowledge Base for fish health screening.

IMPORTANT:
This knowledge base provides educational context and general biosecurity advice only.
It is NOT a veterinary diagnosis or prescription tool.
No specific pharmaceutical drug names or dosages are provided.
"""

KNOWLEDGE_BASE = {
    "Aeromoniasis": {
        "name": "Aeromoniasis (Bacterial Infection / Ulcer Disease)",
        "is_disease": True,
        "description": "A common bacterial infection in freshwater and cultured fish often linked to environmental stress or poor water conditions.",
        "visible_signs": [
            "Hemorrhagic spots or reddening on skin, fins, or vent",
            "Open skin ulcers or erosions",
            "Abdominal distension (dropsy) or raised scales",
            "Lethargy or loss of appetite"
        ],
        "general_guidance": [
            "Immediately isolate showing fish in a dedicated quarantine tank.",
            "Test water parameters (ammonia, nitrite, nitrate, pH, dissolved oxygen) and perform partial water exchanges if indicated.",
            "Reduce feed rations temporarily and minimize handling stress.",
            "Consult a licensed aquaculture veterinarian or aquatic animal health professional for diagnostic culture and appropriate treatment."
        ]
    },
    "branchitis": {
        "name": "Branchitis (Gill Inflammation / Gill Disease)",
        "is_disease": True,
        "description": "Inflammation or irritation of gill tissues caused by environmental irritants, poor water quality, or pathogenic agents.",
        "visible_signs": [
            "Swollen, pale, mottled, or excessively mucus-covered gill filaments",
            "Rapid or labored breathing (flaring opercula)",
            "Piping at the water surface or crowding near aeration inlets",
            "Reduced activity and reluctance to feed"
        ],
        "general_guidance": [
            "Immediately check and increase aeration; ensure dissolved oxygen levels remain optimal.",
            "Verify ammonia and nitrite levels immediately, as toxic spikes severely irritate gill membranes.",
            "Ensure gentle water flow and avoid suspended solids or chemical irritants in the tank.",
            "Seek guidance from an aquaculture specialist to identify underlying causes (parasitic, bacterial, or environmental)."
        ]
    },
    "keratitis": {
        "name": "Keratitis (Corneal Cloudiness / Eye Inflammation)",
        "is_disease": True,
        "description": "Inflammation or opacity affecting the cornea or eye structure, frequently triggered by physical abrasion, bright lighting, or secondary infections.",
        "visible_signs": [
            "Cloudy, milky, or opaque appearance over one or both eyes",
            "Swollen or protruding eyes (exophthalmia)",
            "Difficulty targeting food during feeding",
            "Rubbing against tank surfaces or substrate"
        ],
        "general_guidance": [
            "Isolate affected fish to prevent secondary tank-mate aggression or injury.",
            "Inspect physical environment for sharp tank decorations, rough net damage, or abrasive substrates.",
            "Maintain pristine water conditions to allow natural corneal healing and prevent secondary bacterial invasion.",
            "Consult an aquatic veterinarian for evaluation if cloudiness worsens or does not clear within a few days."
        ]
    },
    "lymphocystis": {
        "name": "Lymphocystis (Viral Skin Lesions)",
        "is_disease": True,
        "description": "A chronic, usually self-limiting viral infection causing hypertrophied dermal cells in freshwater and marine fish.",
        "visible_signs": [
            "Nodular, cauliflower-like, or wart-like growths on fins, mouth, and skin",
            "White, pinkish, or cream-colored firm clusters",
            "Lesions usually slow-growing without deep ulceration"
        ],
        "general_guidance": [
            "Isolate symptomatic fish to reduce shedding and viral transmission within the population.",
            "Provide optimal nutrition and strictly stable water quality to bolster the fish's natural immune response.",
            "Avoid attempting manual removal of nodules, as this risks deep tissue injury and secondary infections.",
            "Consult an aquatic animal health advisor to confirm viral etiology and establish monitoring protocols."
        ]
    },
    "object": {
        "name": "Reference Object / Non-Disease Region",
        "is_disease": False,
        "description": "Foreground object, tag, measurement marker, or non-pathological visual reference detected in the image.",
        "visible_signs": [
            "Artificial tag, scale marker, background artifact, or non-symptomatic anatomical feature"
        ],
        "general_guidance": [
            "This detection is categorized as non-disease.",
            "No health management action required for this visual element."
        ]
    }
}
