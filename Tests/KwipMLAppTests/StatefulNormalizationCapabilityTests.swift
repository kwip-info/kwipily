import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class StatefulNormalizationCapabilityTests:XCTestCase {
    private func workspace()throws->WorkspaceStore {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:storage)}
        return WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
    }
    private func execution(_ store:WorkspaceStore)throws->Data {
        try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
    }
    func testPublishedNormalizationWithoutDedicatedDiscoveryRefusesEverySelectedOperator()throws {
        let store=try workspace()
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:execution(store))
        for id in StatefulNormalizationRuntimeCapabilities.operations {
            let graph=try ExampleFlows.statefulNormalizationClassifier(componentID:id)
            let message=store.capabilityBoundaryError(for:graph)
            XCTAssertNotNil(message,id)
            XCTAssertTrue(message?.contains(BundledNormalizationComponents.definition(id).title) == true,message ?? id)
            XCTAssertTrue(message?.contains("verified runtime support") == true,message ?? id)
        }
    }
    func testPublishedNormalizationCannotInflateVerifiedComponentCount()throws {
        let store=try workspace()
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:execution(store))
        // Existing primitives without their dedicated descriptors remain unavailable.
        // Only Input, Output, Linear, ReLU and Tanh are admitted by the base contract.
        XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
    }
    /// Independent explicit discovery records; not encoded from the Core decoder.
    private func descriptor(_ id:String)throws->Data {
        let ranks=["batch_norm1d":[1,2],"batch_norm2d":[3],"instance_norm1d":[2],"instance_norm2d":[3],"group_norm":[1,2,3,4,5,6,7]]
        var value:[String:Any]=["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2."+id,"graphSchemaVersions":[2],"inputSampleRanks":ranks[id]!,"channelAxis":1,"channelMinimum":1,"channelMaximum":512,"spatialMinimum":1,"spatialMaximum":4096,"preservesBatch":true,"graphDTypes":["float32"],"epsilonMinimum":1.1754943508222875e-38,"epsilonMaximum":3.4028234663852886e38,"affineModes":[false,true],"biasModes":[false,true],"forwardVarianceCorrection":0,"maximumTensorElements":100000000]
        if id == "group_norm" {
            value.merge(["groupMinimum":1,"groupMaximum":512,"requiresChannelDivisibility":true,"minimumGroupMembers":2,"statisticsScope":"observationGroupSpatial","usesCurrentStatistics":true]) {$1}
        } else {
            value.merge(["momentumMinimum":0.0,"momentumMaximum":1.0,"trackingModes":[false,true],"runningVarianceCorrection":1,"runningMeanDType":"float32","runningVarianceDType":"float32","counterDType":"int64","counterMaximum":1000000000,"statisticsScope":id.hasPrefix("batch") ? "batchSpatial":"observationChannelSpatial","evaluationUsesRunningStatisticsWhenTracked":true]) {$1}
            if id.hasPrefix("batch") {value.merge(["averagingModes":["moving","cumulative"],"counterBehavior":"trainingCalls","minimumCurrentStatisticsMembers":2]) {$1}}
            else {value.merge(["counterBehavior":"constantZero","minimumCurrentStatisticsSpatialElements":2]) {$1}}
        }
        return try JSONSerialization.data(withJSONObject:value)
    }
    func testExactPerMemberDiscoveryGrantsOnlyItsOwnReadinessAndCount()throws {
        let store=try workspace(),contract=try execution(store)
        for id in StatefulNormalizationRuntimeCapabilities.operations {
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,statefulNormalizationDescriptors:[id:descriptor(id)])
            XCTAssertEqual(store.statefulNormalizationRuntimeCapabilities.componentIDs,[id])
            XCTAssertEqual(store.verifiedRuntimeComponentCount,6)
            for other in StatefulNormalizationRuntimeCapabilities.operations where other != id {
                XCTAssertTrue(store.capabilityBoundaryError(for:try ExampleFlows.statefulNormalizationClassifier(componentID:other))?.contains("verified runtime support") == true,other)
            }
            let graph=try ExampleFlows.statefulNormalizationClassifier(componentID:id)
            // The example also requires reshape/flatten discovery; normalization must
            // stop being its readiness blocker without granting unrelated operations.
            XCTAssertFalse(store.capabilityBoundaryError(for:graph)?.contains(BundledNormalizationComponents.definition(id).title+" requires") == true,id)
        }
    }
    func testMalformedStatePolicyAndIdentityNeverGrantSupport()throws {
        let store=try workspace(),contract=try execution(store)
        for id in StatefulNormalizationRuntimeCapabilities.operations {
            for key in ["graphDTypes","inputSampleRanks","adapterID","statePolicy"] {
                var value=try XCTUnwrap(JSONSerialization.jsonObject(with:descriptor(id)) as? [String:Any])
                if key == "graphDTypes" {value[key]=["int64"]}
                else if key == "inputSampleRanks" {value[key]=[8]}
                else if key == "adapterID" {value[key]="unreviewed.adapter"}
                else if id == "group_norm" {value["minimumGroupMembers"]=1}
                else {value["counterBehavior"]="trainable"}
                try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,statefulNormalizationDescriptors:[id:JSONSerialization.data(withJSONObject:value)])
                XCTAssertTrue(store.statefulNormalizationRuntimeCapabilities.componentIDs.isEmpty)
                XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
                XCTAssertTrue(store.capabilityBoundaryError(for:try ExampleFlows.statefulNormalizationClassifier(componentID:id))?.contains("verified runtime support") == true,id)
            }
        }
    }
    func testRefreshAndFailedVersionRevokePreviouslyVerifiedStatefulOperators()throws {
        let store=try workspace(),contract=try execution(store)
        let descriptors=try Dictionary(uniqueKeysWithValues:StatefulNormalizationRuntimeCapabilities.operations.map {($0,try descriptor($0))})
        func install()throws {try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract,statefulNormalizationDescriptors:descriptors)}
        try install();XCTAssertEqual(store.statefulNormalizationRuntimeCapabilities.componentIDs.count,5);XCTAssertEqual(store.verifiedRuntimeComponentCount,10)
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:contract)
        XCTAssertTrue(store.statefulNormalizationRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
        try install();XCTAssertThrowsError(try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8)))
        XCTAssertTrue(store.statefulNormalizationRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
        try install();XCTAssertThrowsError(try store.installRuntimeContract(version:"bad",expectedVersion:nil))
        XCTAssertTrue(store.statefulNormalizationRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,0)
    }
    func testActualWorkerDiscoversAllFiveReviewedDescriptors()async throws {
        let store=try workspace();_ = try XCTUnwrap(store.runtime,store.runtimeError ?? "Actual worker required")
        let ready=await withCheckedContinuation {continuation in store.refreshCapabilities {continuation.resume(returning:$0)}}
        XCTAssertTrue(ready,store.capabilityStatus)
        XCTAssertEqual(store.statefulNormalizationRuntimeCapabilities.componentIDs,StatefulNormalizationRuntimeCapabilities.operations)
        for id in StatefulNormalizationRuntimeCapabilities.operations {
            XCTAssertEqual(store.registry.component(type:id),BundledNormalizationComponents.definition(id))
        }
    }

}
