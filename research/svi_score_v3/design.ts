import {
  GENERATOR_FAMILIES,
  randomizedAuditScenario,
} from "../score_v3/population";
import type { GeneratorFamilyContract } from "../score_v3/types";
import { SVI_SCORE_V3_CONTRACT } from "./contract";

export interface SviDesignBusiness {
  family: GeneratorFamilyContract;
  familyIndex: number;
  scenario: ReturnType<typeof randomizedAuditScenario>;
}

/**
 * The design is deterministic and uses every predeclared generator-family slot.
 * Changing ordering or counts changes the research contract and requires a new
 * version rather than silently changing a running cohort.
 */
export function sviDesignBusinesses(): SviDesignBusiness[] {
  const businesses = GENERATOR_FAMILIES.flatMap((family) =>
    Array.from({ length: family.businessCount }, (_, familyIndex) => ({
      family,
      familyIndex,
      scenario: randomizedAuditScenario(family, 12_000 + familyIndex),
    })),
  );
  if (businesses.length !== SVI_SCORE_V3_CONTRACT.designBusinesses) {
    throw new Error(
      `SVI design expected ${SVI_SCORE_V3_CONTRACT.designBusinesses} businesses but generated ${businesses.length}.`,
    );
  }
  return businesses;
}

export function stratifiedSviDesignSubset(count: number): SviDesignBusiness[] {
  const all = sviDesignBusinesses();
  if (count >= all.length) return all;
  const queues = GENERATOR_FAMILIES.map((family) =>
    all.filter((business) => business.family.id === family.id),
  );
  const selected: SviDesignBusiness[] = [];
  for (let index = 0; selected.length < count; index += 1) {
    for (const queue of queues) {
      const next = queue[index];
      if (next) selected.push(next);
      if (selected.length === count) break;
    }
  }
  return selected;
}
