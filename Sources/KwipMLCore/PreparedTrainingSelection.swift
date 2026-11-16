import Foundation

/// Private reference to an immutable prepared-data run. Runtime code must verify the
/// bounded manifest bytes against this hash before trusting its endpoint artifacts.
public struct PreparedTrainingSelection:Codable,Equatable {
    public var namedInputPlan:NamedInputPlan? = nil
    public var manifestPath:String
    public var manifestSHA256:String
    public var graphDigest:String
    public var trainingNode:String
    public var trainingPort:String
    public var validationNode:String
    public var validationPort:String
    public var featureColumns:[String]
    public var targetColumn:String
    public init(manifestPath:String,manifestSHA256:String,graphDigest:String,trainingNode:String,trainingPort:String,validationNode:String,validationPort:String,featureColumns:[String],targetColumn:String,namedInputPlan:NamedInputPlan?=nil){
        self.namedInputPlan=namedInputPlan
        self.manifestPath=manifestPath;self.manifestSHA256=manifestSHA256;self.graphDigest=graphDigest
        self.trainingNode=trainingNode;self.trainingPort=trainingPort;self.validationNode=validationNode;self.validationPort=validationPort
        self.featureColumns=featureColumns;self.targetColumn=targetColumn
    }
    public var reference: PreparedDataReference {
        .init(manifestPath: manifestPath, manifestSHA256: manifestSHA256, graphDigest: graphDigest,
              trainingNode: trainingNode, trainingPort: trainingPort,
              validationNode: validationNode, validationPort: validationPort)
    }
    public func validate()throws {
        try namedInputPlan?.validate()
        try reference.validate()
        try TrainingColumnMapping.validate(features:featureColumns,target:targetColumn)
    }
}
