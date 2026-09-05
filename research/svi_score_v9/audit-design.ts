import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../score_v3/population";
import type { GeneratorFamilyContract } from "../score_v3/types";
import {
  SVI_SCORE_V9_AUDIT_CONTRACT,
  V9_AUDIT_FAMILY_IDS,
} from "./audit-contract";

export interface V9AuditBusiness {
  family: GeneratorFamilyContract;
  sourceSplit: GeneratorFamilyContract["split"];
  auditFamilyIndex: number;
  scenario: ReturnType<typeof randomizedAuditScenario>;
}

export function v9SealedAuditBusinesses(): V9AuditBusiness[] {
  const byId = new Map(GENERATOR_FAMILIES.map((family) => [family.id, family]));
  const businesses = V9_AUDIT_FAMILY_IDS.flatMap((familyId) => {
    const source = byId.get(familyId);
    if (!source) throw new Error(`Unknown V9 audit generator family: ${familyId}.`);
    return Array.from(
      { length: SVI_SCORE_V9_AUDIT_CONTRACT.cohort.businessesPerFamily },
      (_, auditFamilyIndex): V9AuditBusiness => {
        const seedIndex = SVI_SCORE_V9_AUDIT_CONTRACT.cohort.seedIndexStart +
          auditFamilyIndex;
        const generated = randomizedAuditScenario(source, seedIndex);
        return {
          family: { ...source, split: "audit", businessCount: 20 },
          sourceSplit: source.split,
          auditFamilyIndex,
          scenario: {
            ...generated,
            id: `v9-audit003-${source.id}-${String(auditFamilyIndex + 1).padStart(2, "0")}`,
            label: `V9 sealed audit 003 · ${source.label} ${auditFamilyIndex + 1}`,
          },
        };
      },
    );
  });
  if (
    businesses.length !== SVI_SCORE_V9_AUDIT_CONTRACT.cohort.businesses ||
    new Set(businesses.map(({ scenario }) => scenario.id)).size !== businesses.length ||
    new Set(businesses.map(({ scenario }) => scenario.seed)).size !== businesses.length
  ) {
    throw new Error("V9 sealed-audit design is incomplete or has duplicate identities.");
  }
  return businesses;
}
