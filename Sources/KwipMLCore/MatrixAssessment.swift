import Foundation

public enum MatrixLearningTask:String,CaseIterable,Codable {
    case contraction = "matrix-contraction"
    case broadcast = "matrix-broadcast-gradients"
    public var title:String {self == .contraction ? "Matrix products · rows and columns":"Matrix products · shared heads and gradients"}
    public var task:String {
        let base="A matrix is a table of numbers arranged in rows and columns. Add the two supplied Inputs, Matrix multiply and Model Output. Connect the left table to a, the right table to b, and the result to Model Output. Each result cell multiplies matching numbers from a left row and a right column, then adds the products. Match the left table’s final size to the right table’s next-to-last size. Keep observations separate. Do not transpose, scale or normalize the tables. Choose Check construction to inspect measured results and change sensitivity: how a small change to either input affects the result. KwipML checks this automatically; no code or handwritten derivatives are required."
        return base+(self == .broadcast ? " Additional grouping directions may share a table when their sizes are equal or size 1. This reuse is called broadcasting. Every reused value collects the effects of all its uses, while the measured influence keeps the original input shape. Observations never share values with one another.":" Match each observation’s row and column calculations. The checker uses other compatible table sizes as well as the representative shapes shown on the canvas.")
    }
}

public enum MatrixAssessment {
    public static func plan(task:MatrixLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int])->LearningTensor {
            let values=(0..<shape.reduce(1,*)).map {_ -> Double in
                state=state &* 6364136223846793005 &+ 1442695040888963407
                return Double(Int((state >> 32)%25)-12)/4
            }
            return .init(shape:shape,values:values)
        }
        let shapes:[([Int],[Int])]=task == .contraction
            ? [([1,2,3],[1,3,4]),([2,3,2],[2,2,3]),([2,1,4],[2,4,1]),([3,2,2],[3,2,2])]
            : [([2,2,3,4],[2,4,2]),([2,3,4],[2,2,4,2]),([2,2,1,3,4],[2,3,4,2]),([2,1,3,4],[2,3,4,2])]
        let fixtures=try shapes.enumerated().map { index,shapes -> LearningFixture in
            let left=tensor(shapes.0),right=tensor(shapes.1)
            let reference=try MatrixLessonReference(left:left,right:right)
            let upstream=tensor(reference.output.shape)
            let gradients=try reference.gradients(for:upstream)
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:["left":left,"right":right,"upstream":upstream],expected:["output":reference.output,"leftGradient":gradients.left,"rightGradient":gradients.right])
        }
        func criterion(_ key:String,_ description:String,_ next:String)->LearningCriterion {
            .init(id:key,description:description,nextStep:next,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))
        }
        let criteria=[
            criterion("output","Multiply matching row and column entries, then add them, separately for each observation.","Check the contraction width and the order of the two operands. Align broadcast sample axes after the observation axis."),
            criterion("leftGradient","Check how small changes to the left input affect the result, keeping its original shape.","Each A[i,k] receives Σⱼ G[i,j]B[k,j]. Sum repeated uses across broadcast axes, never across observations."),
            criterion("rightGradient","Check how small changes to the right input affect the result, including reused values.","Each B[k,j] receives Σᵢ A[i,k]G[i,j]. Broadcast reuses values; reduce those uses back to B's original shape.")
        ]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"linear-algebra."+task.rawValue,task:task.task,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["matmul":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate();return plan
    }
}
