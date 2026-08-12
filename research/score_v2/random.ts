export interface SeededRandom {
  uniform(): number;
  normal(): number;
}

export function createSeededRandom(seed: number): SeededRandom {
  let state = seed >>> 0;
  let spareNormal: number | undefined;

  const uniform = () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };

  const normal = () => {
    if (spareNormal !== undefined) {
      const value = spareNormal;
      spareNormal = undefined;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1e-12)));
    const angle = 2 * Math.PI * uniform();
    spareNormal = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };

  return { uniform, normal };
}
