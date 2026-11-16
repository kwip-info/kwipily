import Foundation

/// Independent bounded scalar oracle. Learner outputs and PyTorch are not references.
public struct BilinearLessonReference {
    public let output: LearningTensor
    private let left: LearningTensor
    private let right: LearningTensor
    private let weight: LearningTensor
    private let bias: LearningTensor?
    private let rows: Int
    private let first: Int
    private let second: Int
    private let outputs: Int

    public init(left: LearningTensor, right: LearningTensor, weight: LearningTensor, bias: LearningTensor?) throws {
        guard left.wellFormed, right.wellFormed, weight.wellFormed,
              left.shape.count >= 2, left.shape.count <= 8,
              left.shape.dropLast().elementsEqual(right.shape.dropLast()), weight.shape.count == 3,
              left.shape.last == weight.shape[1], right.shape.last == weight.shape[2] else {
            throw LearningContractError("Bilinear needs matching observation/sample prefixes and weights [output, first features, second features]. No axes broadcast.")
        }
        let first = weight.shape[1], second = weight.shape[2], outputs = weight.shape[0]
        let rows = left.values.count / first
        guard rows <= 4096 / outputs, weight.values.count <= 100_000 / rows else {
            throw LearningContractError("Bilinear teaching fixtures allow at most 4096 outputs and 100000 scalar interaction terms.")
        }
        if let bias {
            guard bias.wellFormed, bias.shape == [outputs] else {
                throw LearningContractError("Bias must contain one finite value per output feature.")
            }
        }
        var values = Array(repeating: 0.0, count: rows * outputs)
        for row in 0..<rows {
            for out in 0..<outputs {
                var value = bias?.values[out] ?? 0
                for i in 0..<first {
                    for j in 0..<second {
                        value += left.values[row*first+i] * weight.values[(out*first+i)*second+j] * right.values[row*second+j]
                    }
                }
                values[row*outputs+out] = value
            }
        }
        guard values.allSatisfy(\.isFinite) else { throw LearningContractError("Bilinear reference overflowed finite values.") }
        self.left=left; self.right=right; self.weight=weight; self.bias=bias
        self.rows=rows; self.first=first; self.second=second; self.outputs=outputs
        output = .init(shape: Array(left.shape.dropLast()) + [outputs], values: values)
    }

    public func gradients(for upstream: LearningTensor) throws -> (left: LearningTensor, right: LearningTensor, weight: LearningTensor, bias: LearningTensor?) {
        guard upstream.wellFormed, upstream.shape == output.shape else {
            throw LearningContractError("Upstream gradient must match the output shape and contain finite values.")
        }
        var da = Array(repeating:0.0,count:left.values.count)
        var db = Array(repeating:0.0,count:right.values.count)
        var dw = Array(repeating:0.0,count:weight.values.count)
        var dc = Array(repeating:0.0,count:outputs)
        for row in 0..<rows {
            for out in 0..<outputs {
                let g = upstream.values[row*outputs+out]; dc[out] += g
                for i in 0..<first {
                    for j in 0..<second {
                        let a=row*first+i, b=row*second+j, w=(out*first+i)*second+j
                        da[a] += g*weight.values[w]*right.values[b]
                        db[b] += g*weight.values[w]*left.values[a]
                        dw[w] += g*left.values[a]*right.values[b]
                    }
                }
            }
        }
        guard [da,db,dw,dc].allSatisfy({ $0.allSatisfy(\.isFinite) }) else {
            throw LearningContractError("Bilinear gradients overflowed finite values.")
        }
        return (.init(shape:left.shape,values:da),.init(shape:right.shape,values:db),
                .init(shape:weight.shape,values:dw),bias.map { .init(shape:$0.shape,values:dc) })
    }

    public static let explanation = "For each output o, y[o] = ΣᵢΣⱼ a[i]W[o,i,j]b[j] + bias[o]. Each weight slice describes interactions between every pair of input features. Unlike a Linear layer, the result depends on products from two operands; unlike matrix multiplication, the interaction weights are learned. Prefix dimensions match exactly. With upstream G[o], dA[i] = Σₒⱼ G[o]W[o,i,j]B[j], dB[j] = Σₒᵢ G[o]A[i]W[o,i,j]. Shared weight gradients sum G[o]A[i]B[j] over observations and sample positions; bias gradients sum G[o]. Inputs never mix observations. Parameter count is O×H1×H2 plus O when bias is enabled."
}
