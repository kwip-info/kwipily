import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Controlled discovery delivery for admission/queue tests. This is not native
/// compute evidence: actual runner integrations use verifyRuntimeForIntegration.
/// It exercises the existing owned-probe seam and inventory binding, never
/// fabricates a production proof or changes its private initializer.
@MainActor func fixtureInstallDiscoveredRuntimeContract(
    _ workspace:WorkspaceStore, version:String, expectedVersion:String?, devices:[String], descriptor:Data,
    geluDescriptor:Data?=nil, sigmoidDescriptor:Data?=nil, tensorLayoutDescriptor:Data?=nil, softmaxDescriptor:Data?=nil,
    layerNormDescriptor:Data?=nil, matrixMultiplyDescriptor:Data?=nil,
    poolingDescriptors:[String:Data]=[:],tensorCompositionDescriptors:[String:Data]=[:],
    remainingActivationDescriptors:[String:Data]=[:],tensorRoutingDescriptors:[String:Data]=[:],
    spatialPatchDescriptors:[String:Data]=[:],statefulNormalizationDescriptors:[String:Data]=[:],
    lifecycleDescriptor:Data?=nil,initializationDescriptor:Data?=nil,
    gradientPolicyDescriptor:Data?=nil,trainabilityDescriptor:Data?=nil,
    moduleCompositionDescriptor:Data?=nil
)throws {
    // Preserve production invalidation behavior before replaying the controlled probe.
    try workspace.installDiscoveredRuntimeContract(
        version:version,
        expectedVersion:expectedVersion,
        devices:devices,
        descriptor:descriptor,
        geluDescriptor:geluDescriptor,
        sigmoidDescriptor:sigmoidDescriptor,
        tensorLayoutDescriptor:tensorLayoutDescriptor,
        softmaxDescriptor:softmaxDescriptor,
        layerNormDescriptor:layerNormDescriptor,
        matrixMultiplyDescriptor:matrixMultiplyDescriptor,
        poolingDescriptors:poolingDescriptors,
        tensorCompositionDescriptors:tensorCompositionDescriptors,
        remainingActivationDescriptors:remainingActivationDescriptors,
        tensorRoutingDescriptors:tensorRoutingDescriptors,
        spatialPatchDescriptors:spatialPatchDescriptors,
        statefulNormalizationDescriptors:statefulNormalizationDescriptors,
        lifecycleDescriptor:lifecycleDescriptor,
        initializationDescriptor:initializationDescriptor,
        gradientPolicyDescriptor:gradientPolicyDescriptor,
        trainabilityDescriptor:trainabilityDescriptor,
        moduleCompositionDescriptor:moduleCompositionDescriptor)
    // Descriptor-only tests deliberately have no executable runtime.
    guard let runtime=workspace.runtime else {return}
    let actualVersion=runtime.packageVersions["torch"] ?? version
    let base=try BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:actualVersion,availableDevices:[.cpu])
    var components:[String:Any]=[:]
    for (key,data) in [("gelu",geluDescriptor),("sigmoid",sigmoidDescriptor),("tensorLayout",tensorLayoutDescriptor),("softmax",softmaxDescriptor),("layerNorm",layerNormDescriptor),("matrixMultiply",matrixMultiplyDescriptor)] {
        if let data {components[key]=try JSONSerialization.jsonObject(with:data)}
    }
    for dictionary in [poolingDescriptors,tensorCompositionDescriptors,remainingActivationDescriptors,tensorRoutingDescriptors,spatialPatchDescriptors,statefulNormalizationDescriptors] {
        for (key,data) in dictionary {components[key]=try JSONSerialization.jsonObject(with:data)}
    }
    var event:[String:Any]=["event":"capabilities","torchVersion":actualVersion,"devices":devices,"defaultDevice":devices.first ?? "cpu","executionContract":try JSONSerialization.jsonObject(with:JSONEncoder().encode(base)),"componentCapabilities":components]
    for (group,key,data) in [("lifecycleCapabilities","checkpoint_lifecycle",lifecycleDescriptor),("initializationCapabilities","initialization",initializationDescriptor),("gradientPolicyCapabilities","gradient_policy",gradientPolicyDescriptor),("trainabilityCapabilities","trainability",trainabilityDescriptor),("compositionCapabilities","moduleComposition",moduleCompositionDescriptor)] {
        if let data {event[group]=[key:try JSONSerialization.jsonObject(with:data)]}
    }
    let line=String(decoding:try JSONSerialization.data(withJSONObject:event),as:UTF8.self)
    let original=workspace.capabilityLaunch
    defer {workspace.capabilityLaunch=original}
    workspace.capabilityLaunch={_,_,_,_,onLine,onExit in onLine(line);onExit(0,"");return .init(cancel:{})}
    var completed:Bool?
    workspace.refreshCapabilities {completed=$0}
    guard completed==true else {throw WorkflowArchiveError("Controlled discovery fixture failed: "+workspace.capabilityStatus)}
}
