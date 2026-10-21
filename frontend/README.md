# AquaVigil Frontend (Planned)

Interactive web dashboard for AquaVigil built with React, TypeScript, and Vite.

## Planned Features (Phase 3 & 4)
- **Image Upload Zone**: Drag-and-drop fish image uploader with preview.
- **Bounding Box Canvas**: Interactive overlay rendering detected disease lesions directly on the image.
- **Explainability Panel**:
  - Class name and symptom labels
  - Detection confidence meters
  - Bounding box dimension details
- **Aquaculture Health Disclaimer**:
  - Prominent banner indicating that model outputs are automated screening aids and must be validated by an aquatic health professional.
- **Metrics Dashboard**:
  - Model precision, recall, and mAP metrics visualized with Recharts.

## Setup Instructions (When Phase 3 Begins)
```bash
cd frontend
npm create vite@latest ./ -- --template react-ts
npm install
npm run dev
```
