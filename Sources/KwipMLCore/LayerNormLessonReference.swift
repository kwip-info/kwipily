import Foundation

/// Independent, bounded Double-precision teaching oracle; does not invoke the model worker.
public struct LayerNormLessonReference {
    public struct Gradients {
        public let input: LearningTensor
        public let scale: LearningTensor?
        public let bias: LearningTensor?
    }
    public let output: LearningTensor
    private let normalized: [Double]
    private let inverseDeviation: [Double]
    private let normalizedShape: [Int]
    private let scales: [Double]
    private let hasScale: Bool
    private let hasBias: Bool

    public init(input: LearningTensor, normalizedShape: [Int], epsilon: Double,
                scale: LearningTensor? = nil, bias: LearningTensor? = nil) throws {
        guard input.wellFormed, input.shape.count >= 2,
              !normalizedShape.isEmpty, normalizedShape.count < input.shape.count,
              Array(input.shape.suffix(normalizedShape.count)) == normalizedShape,
              LayerNormConfiguration.epsilonRange.contains(epsilon) else {
            throw LearningContractError("Layer normalization requires a finite batched tensor, exact trailing sample dimensions and a positive normal float32 epsilon.")
        }
        for parameter in [scale,bias].compactMap({ $0 }) {
            guard parameter.wellFormed, parameter.shape == normalizedShape else {
                throw LearningContractError("Elementwise scale and bias must match the normalized dimensions exactly.")
            }
        }
        guard bias == nil || scale != nil else {
            throw LearningContractError("Layer normalization bias requires elementwise affine scaling to be enabled.")
        }
        let width = normalizedShape.reduce(1,*) // Exact suffix of the bounded, validated input.
        let scales = scale?.values ?? Array(repeating: 1.0, count: width)
        let biases = bias?.values ?? Array(repeating: 0.0, count: width)
        var normalized = Array(repeating: 0.0, count: input.values.count)
        var result = normalized
        var inverseDeviation: [Double] = []
        for start in stride(from: 0, to: input.values.count, by: width) {
            // Population variance (divide by m), with epsilon inside the square root.
            let mean = (0..<width).reduce(0.0) { $0 + input.values[start+$1]/Double(width) }
            let centered = (0..<width).map { input.values[start+$0]-mean }
            let variance = centered.reduce(0.0) { $0 + $1*$1/Double(width) }
            let inverse = 1 / sqrt(variance + epsilon)
            guard mean.isFinite, variance.isFinite, inverse.isFinite else {
                throw LearningContractError("Normalization reference statistics exceeded finite numeric range.")
            }
            inverseDeviation.append(inverse)
            for offset in 0..<width {
                let index = start+offset
                normalized[index] = centered[offset]*inverse
                result[index] = normalized[index]*scales[offset]+biases[offset]
            }
        }
        guard result.allSatisfy(\.isFinite), normalized.allSatisfy(\.isFinite) else {
            throw LearningContractError("Normalization reference output exceeded finite numeric range.")
        }
        self.output = LearningTensor(shape: input.shape, values: result)
        self.normalized = normalized; self.inverseDeviation = inverseDeviation
        self.normalizedShape = normalizedShape; self.scales = scales
        self.hasScale = scale != nil; self.hasBias = bias != nil
    }

    public func gradients(for upstream: LearningTensor) throws -> Gradients {
        guard upstream.wellFormed, upstream.shape == output.shape else {
            throw LearningContractError("Upstream gradients must match the normalized output shape and contain finite values.")
        }
        let width = scales.count
        var input = Array(repeating: 0.0, count: normalized.count)
        var scale = Array(repeating: 0.0, count: width)
        var bias = scale
        for (group,start) in stride(from: 0, to: input.count, by: width).enumerated() {
            let h = (0..<width).map { upstream.values[start+$0]*scales[$0] }
            let meanH = h.reduce(0,+)/Double(width)
            let meanHZ = (0..<width).reduce(0.0) { $0+h[$1]*normalized[start+$1] }/Double(width)
            for offset in 0..<width {
                let index = start+offset
                input[index] = inverseDeviation[group]*(h[offset]-meanH-normalized[index]*meanHZ)
                if hasScale { scale[offset] += upstream.values[index]*normalized[index] }
                if hasBias { bias[offset] += upstream.values[index] }
            }
        }
        guard input.allSatisfy(\.isFinite), scale.allSatisfy(\.isFinite), bias.allSatisfy(\.isFinite) else {
            throw LearningContractError("Normalization reference gradients exceeded finite numeric range.")
        }
        return Gradients(input: .init(shape: output.shape, values: input),
                         scale: hasScale ? .init(shape: normalizedShape, values: scale) : nil,
                         bias: hasBias ? .init(shape: normalizedShape, values: bias) : nil)
    }

    public static let explanation = "For each trailing sample slice of m values, compute μ = Σxᵢ/m and v = Σ(xᵢ−μ)²/m. Normalize zᵢ = (xᵢ−μ)/√(v+ε), then optionally learn yᵢ = γᵢzᵢ+βᵢ. This uses population variance, not the m−1 sample estimate. Epsilon is inside the square root; the normalized variance is v/(v+ε), so it need not equal one. Scale and bias have one parameter per trailing coordinate, shared across preceding axes and observations. Both training and evaluation use current-input statistics. For upstream g, let h = g⊙γ (or g without scale); dL/dx = (h−mean(h)−z mean(h⊙z))/√(v+ε). Scale gradients sum g⊙z and bias gradients sum g across shared slices. Observations remain separate; there are no running dataset statistics."
}
