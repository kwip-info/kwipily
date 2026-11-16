import Foundation
import KwipMLCore

/// Checks captured policy and native ownership metadata; tensor bytes remain the worker's strict responsibility.
enum WorkflowInitializationVerifier {
    static func validate(_ manifest:[String:JSONValue],expected:InitializationConfiguration?,graph:GraphDocument,
                         trainingMode:WorkflowTrainingMode,context:GraphValidationContext = .preparedWorkflow,
                         capturedRegistry:ComponentRegistry?=nil)throws {
        func fail()->WorkflowPhaseError {WorkflowPhaseError("Starting-settings artifacts differ from the captured recipe, actual learned owners, generator policy or training mode.")}
        func require(_ valid:Bool)throws {guard valid else {throw fail()}}
        guard case .object(let settings)=manifest["settings"],case .object(let provenance)=manifest["provenance"] else {throw fail()}
        guard let expected else {
            var unrequested:JSONValue?
            if case .object(let checkpoint)=manifest["checkpoint"],case .object(let contract)=checkpoint["contract"] {unrequested=contract["initialization"]}
            try require(settings["initializationRecipe"] == nil && provenance["initialization"] == nil && unrequested == nil)
            return
        }
        guard case .object(let checkpoint)=manifest["checkpoint"],case .object(let contract)=checkpoint["contract"],
              let rawRecipe=settings["initializationRecipe"],
              case .object(var identity)=contract["initialization"],
              case .object(var actual)=provenance["initialization"],
              case .object(let runtime)=manifest["runtime"],
              let version=runtime["torch"]?.stringValue,!version.isEmpty,version.utf8.count<=128,
              case .object(let pinnedRuntime)=contract["runtime"],
              case .object(let lineage)=manifest["lineage"] else {throw fail()}
        try require(runtime["device"] == .string("cpu") || runtime["device"] == .string("mps"))
        try require(pinnedRuntime["torch"] == .string(version))
        if trainingMode == .exactResume {try require(runtime["device"] == .string("cpu"))}
        let mode=trainingMode == .fresh ? "newTraining":trainingMode == .exactResume ? "exactResume":"weightsOnlyFineTune"
        try require(lineage["mode"] == .string(mode))
        func canonicalRecipe(_ value:JSONValue?)throws->JSONValue {
            guard let value else {throw fail()}
            let configuration=try InitializationConfiguration.decodeRecipe(value)
            try require(configuration == expected)
            // A wire may spell an integral real as 1 or 1.0. Strict typed decoding
            // canonicalizes that representation without accepting unknown controls.
            return configuration.recipeValue
        }
        _ = try canonicalRecipe(rawRecipe)
        identity["recipe"] = try canonicalRecipe(identity["recipe"])
        actual["recipe"] = try canonicalRecipe(actual["recipe"])
        let all=try WorkflowLifecycleVerifier.optimizerParameterOwnership(graph,context:context,capturedRegistry:capturedRegistry,initializationOwnership:true)
        var byOwner:[String:[String:[String:JSONValue]]]=[:],allNames=Set<String>(),total=0
        for item in all {
            guard case .object(let owner)=item,let node=owner["node"]?.stringValue,let parameter=owner["parameter"]?.stringValue,
                  let name=owner["name"]?.stringValue,allNames.insert(name).inserted,
                  case .array(let shape)=owner["shape"],owner["dtype"] == .string("float32") else {throw fail()}
            var count=1
            for axis in shape {
                guard case .integer(let extent)=axis,extent>=1,extent<=200_000,count<=200_000/Int(extent) else {throw fail()}
                count *= Int(extent)
            }
            try require(total<=200_000-count && byOwner[node]?[parameter] == nil)
            total += count
            byOwner[node,default:[:]][parameter]=owner
        }
        var selected:[JSONValue]=[]
        for rule in expected.rules {
            guard let owner=byOwner[rule.owner.node]?[rule.owner.parameter],case .array(let shape)=owner["shape"] else {throw fail()}
            if rule.algorithm == "xavier_uniform" || rule.algorithm == "kaiming_uniform" {try require(shape.count>=2)}
            selected.append(.object(owner))
        }
        let wanted:[String:JSONValue]=["schemaVersion":1,"recipe":expected.recipeValue,"resolvedOwners":.array(selected),
            "generatorPlacement":.string("cpu-float32"),"seedDerivation":.string("kwip.initialization.owner-seed.v1")]
        try require(identity == wanted)
        var applied=wanted
        applied["torchVersion"] = .string(version)
        applied["application"] = .string(trainingMode == .fresh ? "applied":"notAppliedOnRestore")
        try require(actual == applied)
    }
}
