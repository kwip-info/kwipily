import Foundation

/// Independent row/column teaching oracle, bounded separately from production execution.
/// No PyTorch execution or learner-supplied results are used to derive reference values.
public struct MatrixLessonReference {
    public let output:LearningTensor
    private let left:LearningTensor
    private let right:LearningTensor
    private let terms:[(output:Int,left:Int,right:Int)]

    public init(left:LearningTensor,right:LearningTensor)throws {
        guard left.wellFormed,right.wellFormed,left.shape.count>=3,right.shape.count>=3,
              left.shape[0] == right.shape[0] else {
            throw LearningContractError("Matrix lessons require finite batched matrices with matching observations.")
        }
        let a=try BatchedTensorLayout(sample:Array(left.shape.dropFirst()))
        let b=try BatchedTensorLayout(sample:Array(right.shape.dropFirst()))
        let result=try a.matrixMultiplied(by:b)
        let shape=[left.shape[0]]+result.sample.dimensions
        let count=left.shape[0]*result.sample.elementCount
        let contraction=left.shape.last!
        guard count<=4096,contraction<=100_000/count else {
            throw LearningContractError("Matrix teaching examples require at most 4096 output values and 100000 scalar products.")
        }
        let rank=shape.count
        let paddedA=[left.shape[0]]+Array(repeating:1,count:rank-left.shape.count)+Array(left.shape.dropFirst())
        let paddedB=[right.shape[0]]+Array(repeating:1,count:rank-right.shape.count)+Array(right.shape.dropFirst())
        func offset(_ coordinates:[Int],_ dimensions:[Int])->Int {
            zip(coordinates,dimensions).reduce(0) {$0*$1.1+($1.1 == 1 ? 0:$1.0)}
        }
        var terms:[(output:Int,left:Int,right:Int)]=[]
        var values=Array(repeating:0.0,count:count)
        for index in 0..<count {
            var remainder=index,coordinates=Array(repeating:0,count:rank)
            for axis in (0..<rank).reversed() {coordinates[axis]=remainder%shape[axis];remainder/=shape[axis]}
            for k in 0..<contraction {
                var ac=coordinates,bc=coordinates
                ac[rank-1]=k;bc[rank-2]=k
                let ai=offset(ac,paddedA),bi=offset(bc,paddedB)
                terms.append((index,ai,bi))
                values[index]+=left.values[ai]*right.values[bi]
            }
        }
        guard values.allSatisfy(\.isFinite) else {throw LearningContractError("Matrix reference output exceeded finite numeric range.")}
        self.left=left;self.right=right;self.terms=terms
        output=LearningTensor(shape:shape,values:values)
    }
    public func gradients(for upstream:LearningTensor)throws->(left:LearningTensor,right:LearningTensor) {
        guard upstream.wellFormed,upstream.shape == output.shape else {
            throw LearningContractError("Upstream gradient must match the matrix product shape and contain finite values.")
        }
        var da=Array(repeating:0.0,count:left.values.count),db=Array(repeating:0.0,count:right.values.count)
        for term in terms {
            da[term.left]+=upstream.values[term.output]*right.values[term.right]
            db[term.right]+=upstream.values[term.output]*left.values[term.left]
        }
        guard da.allSatisfy(\.isFinite),db.allSatisfy(\.isFinite) else {
            throw LearningContractError("Matrix reference gradients exceeded finite numeric range.")
        }
        return (.init(shape:left.shape,values:da),.init(shape:right.shape,values:db))
    }
    public static let explanation="A is the left table, B is the right table, and C is the resulting table. Indices i and j select a result row and column; k selects a matching position along the shared width. Each output C[i,j] = Σₖ A[i,k]B[k,j]: multiply the paired entries and sum them over k. The shared width is called the contraction direction; it disappears from the result after summation. G is the supplied importance of each result cell when measuring change sensitivity. The symbol dA means the combined influence on the left table, and dB means that influence on the right table. A superscript T means transpose: swap the final row and column directions. Thus dA = GBᵀ and dB = AᵀG. When a size-1 grouping direction broadcasts, its table is reused; the influence sums every reuse back into the original input shape. The observation direction never broadcasts. In attention, Q names query vectors and K names key vectors: QKᵀ compares every query with every key. Transpose the key’s final two directions explicitly; matrix multiplication alone neither scales nor normalizes those comparisons. These equations explain the automatic checks; learners do not need to enter derivatives."
}
