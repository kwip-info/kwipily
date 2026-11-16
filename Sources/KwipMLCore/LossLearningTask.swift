import Foundation

public enum LossLearningTask:String,CaseIterable,Codable {
    case crossEntropy="cross-entropy-class-choice",bce="bce-independent-labels",mse="mse-error-size",l1="l1-absolute-error",nll="nll-log-probability-choice",kl="kl-distribution-comparison",ctc="ctc-sequence-alignment",triplet="triplet-relative-distance"
    public var componentID:String {switch self {case .crossEntropy:return "loss_cross_entropy";case .bce:return "loss_bce_with_logits";case .mse:return "loss_mse";case .l1:return "loss_l1";case .nll:return "loss_nll";case .kl:return "loss_kl_div";case .ctc:return "loss_ctc";case .triplet:return "loss_triplet_margin"}}
    public var configuration:[String:JSONValue] {
        switch self {
        case .crossEntropy:return ["reduction":.string("mean"),"class_weights":.array([1,2,0.5]),"label_smoothing":0.2]
        case .bce:return ["reduction":.string("none"),"weight_mode":.string("mapped"),"pos_weight_mode":.string("configured"),"pos_weight_shape":.array([2]),"pos_weight_values":.array([2,3])]
        case .mse:return ["reduction":.string("sum"),"target_broadcast":true]
        case .l1:return ["reduction":.string("none")]
        case .nll:return ["class_weights":.array([1,2,0.5])]
        case .kl:return ["reduction":.string("batchmean"),"log_target":true]
        case .ctc:return ["zero_infinity":true,"target_layout":.string("packed")]
        case .triplet:return ["margin":0.8,"p":1.5,"eps":0,"swap":true]
        }
    }
    public var predictionSampleShape:[Int] {self == .ctc ? [3,3]:self == .triplet ? [3,2]:[self == .crossEntropy || self == .nll || self == .kl ? 3:2]}
    public var sourceRoles:[String] {self == .triplet ? ["prediction"]:self == .ctc ? ["prediction","target","input_lengths","target_lengths"]:self == .bce ? ["prediction","target","weight"]:["prediction","target"]}
    public var usesLogSoftmax:Bool {self == .nll || self == .kl || self == .ctc}
    public var title:String {
        switch self {
        case .crossEntropy:return "CrossEntropyLoss · Choose a class"
        case .bce:return "BCEWithLogitsLoss · Judge independent labels"
        case .mse:return "MSELoss · Give larger mistakes more importance"
        case .l1:return "L1Loss · Measure the distance from a target"
        case .nll:return "NLLLoss · Use prepared class probabilities"
        case .kl:return "KLDivLoss · Compare two distributions"
        case .ctc:return "CTCLoss · Match a sequence without position labels"
        case .triplet:return "TripletMarginLoss · Bring matching examples closer"
        }
    }
    public var instructions:String {
        let controls:String,meaning:String
        switch self {
        case .crossEntropy:controls="Set Class importance to 1, 2, 0.5; Smoothing share to 0.2; Combine scores to Mean.";meaning="A class is one possible answer. A score compares the choices before turning them into probabilities. Smoothing reserves a small share for every choice, so the desired answer is less absolute. Class importance changes how strongly a mistake counts."
        case .bce:controls="Set Element importance source to Data connection. Set Positive-label importance source to Configured values, shape to 2 and values to 2, 3. Choose Keep individual scores.";meaning="Independent labels can both be true, unlike choosing one class. A desired share between 0 and 1 expresses uncertainty. Element importance changes a whole score; positive-label importance changes only the part for a desired positive answer."
        case .mse:controls="Enable Reuse target directions of size 1 and choose Sum.";meaning="Squared error multiplies each difference by itself. A mistake twice as large counts four times as much. One desired value is deliberately reused across each observation's two values; observations are never mixed."
        case .l1:controls="Choose Keep individual scores and leave Reuse target directions of size 1 disabled.";meaning="Absolute error is the distance between the produced and desired values, without a negative sign. A mistake twice as large counts twice as much."
        case .nll:controls="Set Class importance to 1, 2, 0.5 and choose Mean.";meaning="A log-probability records a probability on a compressed number scale: probability 1 becomes 0, and smaller probabilities become negative numbers. It is not the probability itself; you do not need to calculate logarithms here. LogSoftmax converts the supplied class scores into this format. A less likely desired class gets a larger loss; the block does not repeat normalization invisibly."
        case .kl:controls="Enable Targets are log-probabilities and choose Total per observation.";meaning="A distribution gives a share to every possible answer. We compare produced shares with desired shares. The targets already use log-probabilities: a compressed number scale where probability 1 becomes 0 and smaller probabilities become negative numbers. These values are not probabilities themselves; LogSoftmax visibly prepares the produced scores. Total per observation adds the class contributions before averaging observations."
        case .ctc:controls="Keep Blank class at 0. Enable Mask impossible alignments and choose packed for Target sequence layout; choose Mean.";meaning="An alignment says which time position produced a desired label. This task supplies labels without those positions. Blank means no new label; consecutive repeats merge unless separated by blank. Input lengths and target lengths say how many positions belong to each observation. Packed targets place those used labels together in one list; target lengths mark where each observation's labels end. A repeat that cannot fit is explicitly masked and contributes zero; that is not a successful match."
        case .triplet:controls="Set Required distance gap to 0.8, Distance power to 1.5 and Distance cushion to 0. Enable Compare matching example with nonmatch too; choose Mean.";meaning="An anchor is the example being compared. Its matching example should be closer than its nonmatching example by the required gap. The three array positions are anchor, match and nonmatch in that order. Distance power changes how directions contribute; the comparison switch also checks the match-to-nonmatch distance."
        }
        return "Build: use the supplied Input roles and one "+title.components(separatedBy:" · ")[0]+" recipe. Connect its out to Objective check.objective. Connect each supplied target or importance Input to its matching Objective check port. "+(usesLogSoftmax ? "Connect prediction through the visible LogSoftmax block, with Normalization direction −1, to Objective check.prediction. ":"Connect prediction directly to Objective check.prediction. ")+"Connect the three Outputs for Individual scores, Combined score and Input influence. \n\nConfigure: "+controls+" On Objective check choose Mean for combining kept scores.\n\nCheck: choose Check construction and compare four changed fixture tables.\n\nUnderstanding: "+meaning+" A loss is a score for how far a result is from its desired outcome. Mean averages scores; Sum adds them; keeping individual scores preserves each cell before Objective check combines them. Class positions start at 0; direction −1 means the last direction. Input influence measures how a small change to an input changes the combined score; it is also called a gradient. You build and inspect blocks; no code or derivatives are required."
    }
    public func fixtures(seed:Int)->[LossLessonFixture] {
        var state=UInt64(bitPattern:Int64(seed))
        return (0..<4).map {index in
            let count=2*predictionSampleShape.reduce(1,*)
            let prediction=(0..<count).map {position->Double in
                state=state &* 6364136223846793005 &+ 1442695040888963407
                let variation=Double(Int((state>>32)%1001)-500)/10000
                return Double((position*3+index)%11-5)*0.13+variation
            }
            var inputs:[String:LearningTensor] = ["prediction":.init(shape:[2]+predictionSampleShape,values:prediction)]
            switch self {
            case .crossEntropy,.nll:inputs["target"] = .init(shape:[2],values:index == 3 ? [0,-100]:[0,2])
            case .bce:inputs["target"] = .init(shape:[2,2],values:[0,1,0.25,0.75]);inputs["weight"] = .init(shape:[2,2],values:[0.5,2,1,0])
            case .mse:inputs["target"] = .init(shape:[2],values:[0.2,-0.4])
            case .l1:inputs["target"] = .init(shape:[2,2],values:[0.2,-0.4,0.5,-0.1])
            case .kl:inputs["target"] = .init(shape:[2,3],values:[0.2,0.5,0.3,0.4,0.2,0.4].map(log))
            case .ctc:inputs["target"] = .init(shape:[2,2],values:[1,2,1,1]);inputs["input_lengths"] = .init(shape:[2],values:[3,2]);inputs["target_lengths"] = .init(shape:[2],values:[2,2])
            case .triplet:break
            }
            return .init(id:rawValue+".seed\(seed).fixture\(index)",inputs:inputs)
        }
    }

}
public struct LossLessonFixture:Codable,Equatable,Identifiable {
    public let id:String
    public let inputs:[String:LearningTensor]
    public init(id:String,inputs:[String:LearningTensor]) {self.id=id;self.inputs=inputs}
}
