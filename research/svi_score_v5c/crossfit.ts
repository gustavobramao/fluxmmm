import { createHash } from "node:crypto";
import type { SviScoreV4CandidateRow } from "../svi_score_v4/types";
import { v5bCandidateUniverse, v5bBaseCandidateId } from "../svi_score_v5b/embedding";
import { v5bMaximinOrder } from "../svi_score_v5b/search";
import {
  SVI_SCORE_V5C_CONTRACT,
  V5C_TRAINING_CONFIGURATION,
} from "./contract";
import { latentV5CScore, learnScoreV5C } from "./ranker";
import type {
  V5CScoreDistribution,
  V5CSelectorEnsemble,
} from "./types";

function average(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) /
    Math.max(values.length, 1);
}

export function v5cCandidateRegions(
  candidateIds: readonly string[],
  regionCount = SVI_SCORE_V5C_CONTRACT.development.candidateRegions,
): Record<string, number> {
  const baseIds = [...new Set(candidateIds.map(v5bBaseCandidateId))].sort();
  const representatives = v5bCandidateUniverse(
    baseIds.map((id) => `${id} · experiments-only`),
  );
  const traversal = v5bMaximinOrder(representatives);
  const regionByBase = new Map(
    traversal.map((candidate, index) => [
      v5bBaseCandidateId(candidate.candidateId),
      index % regionCount,
    ]),
  );
  return Object.fromEntries(
    [...new Set(candidateIds)].sort().map((candidateId) => [
      candidateId,
      regionByBase.get(v5bBaseCandidateId(candidateId)),
    ]),
  ) as Record<string, number>;
}

function memberAssignments(
  rows: readonly SviScoreV4CandidateRow[],
  memberCount: number,
): Map<string, number> {
  const familyByBusiness = new Map<string, string>();
  rows.forEach((row) => familyByBusiness.set(row.businessId, row.family));
  const families = [...new Set(familyByBusiness.values())].sort();
  const output = new Map<string, number>();
  families.forEach((family) => {
    [...familyByBusiness.entries()]
      .filter(([, value]) => value === family)
      .map(([businessId]) => businessId)
      .sort((left, right) => {
        const leftHash = createHash("sha256").update(left).digest("hex");
        const rightHash = createHash("sha256").update(right).digest("hex");
        return leftHash.localeCompare(rightHash) || left.localeCompare(right);
      })
      .forEach((businessId, index) => output.set(businessId, index % memberCount));
  });
  return output;
}

export function fitV5CSelectorEnsemble(
  rows: readonly SviScoreV4CandidateRow[],
  regionByCandidateId = v5cCandidateRegions(rows.map((row) => row.candidateId)),
): V5CSelectorEnsemble {
  if (!rows.length) throw new Error("V5C ensemble requires training rows.");
  const memberCount = SVI_SCORE_V5C_CONTRACT.development.uncertaintyMembersPerRegion;
  const assignments = memberAssignments(rows, memberCount);
  const regions = Array.from(
    { length: SVI_SCORE_V5C_CONTRACT.development.candidateRegions },
    (_, region) => {
      const excludedCandidateIds = Object.entries(regionByCandidateId)
        .filter(([, candidateRegion]) => candidateRegion === region)
        .map(([candidateId]) => candidateId)
        .sort();
      const eligible = rows.filter(
        (row) => regionByCandidateId[row.candidateId] !== region,
      );
      const models = Array.from({ length: memberCount }, (_, member) => {
        const training = eligible.filter(
          (row) => assignments.get(row.businessId) !== member,
        );
        return learnScoreV5C(training, V5C_TRAINING_CONFIGURATION);
      });
      return {
        region,
        models,
        trainingBusinesses: new Set(eligible.map((row) => row.businessId)).size,
        excludedCandidateIds,
      };
    },
  );
  return {
    regions,
    regionByCandidateId: structuredClone(regionByCandidateId),
  };
}

export function scoreV5CDistribution(
  row: SviScoreV4CandidateRow,
  ensemble: V5CSelectorEnsemble,
): V5CScoreDistribution {
  const region = ensemble.regionByCandidateId[row.candidateId];
  const receipt = ensemble.regions.find((item) => item.region === region);
  if (!receipt) throw new Error(`V5C has no cross-fitted region for ${row.candidateId}.`);
  const members = receipt.models.map((model) => latentV5CScore(row, model));
  const mean = average(members);
  const variance = members.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    Math.max(members.length - 1, 1);
  const standardDeviation = Math.sqrt(Math.max(0, variance));
  return {
    mean,
    standardDeviation,
    robust: mean -
      SVI_SCORE_V5C_CONTRACT.uncertainty.robustScoreStandardErrors *
        standardDeviation,
    members,
  };
}
