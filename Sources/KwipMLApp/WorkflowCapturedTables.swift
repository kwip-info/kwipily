import Foundation
import KwipMLCore

/// Query capabilities admitted from the same authenticated preparation manifest used
/// by training/backend fit. Admission reads metadata only; queries recheck artifact pins.
struct WorkflowCapturedTables {
    let artifacts:[WorkflowEndpoint:TableViewQuerySession.Artifact]
    let images:FileImageArtifactResolver
    let runtime:RuntimeLocation
    let storageLimitBytes:Int64?
    static func admit(bytes:Data,expectedSHA256:String,directory:URL,graph:DataFlowDocument,
                      runtime:RuntimeLocation,storageLimitBytes:Int64?)throws->Self? {
        guard WorkflowArchiveCodec.hash(bytes)==expectedSHA256 else {throw WorkflowArchiveError("Captured table manifest differs from the accepted preparation.")}
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        // Older captures remain readable as bounded observations, without inventing
        // complete-table capabilities absent from their execution coverage.
        guard manifest["executionCoverage"] != nil else {return nil}
        var artifacts=try TableViewArtifactReference.capturedTables(manifest:manifest,directory:directory,expectedNodes:Set(graph.nodes.map(\.id)))
        let views=try TableViewArtifactReference.parse(manifest:manifest,directory:directory,expectedNodes:Set(graph.nodes.filter {$0.operatorID=="kwip.data.view"}.map(\.id)))
        for (node,view) in views {artifacts[.init(address:.init(domain:.data,localID:node),port:"table")]=view}
        struct Inventory:Decodable {let artifacts:[WorkflowArchiveFile]}
        let inventory=try JSONDecoder().decode(Inventory.self,from:bytes)
        return .init(artifacts:artifacts,images:try .init(directory:directory,verifiedArtifacts:inventory.artifacts),runtime:runtime,storageLimitBytes:storageLimitBytes)
    }
}
