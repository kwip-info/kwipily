import Foundation
import KwipMLCore

extension DataFlowWorkspaceStore {
    func availableColumnSchema(for id: String, verifySources: Bool = true, inputPort:String? = nil) -> [DataColumnSchema] {
        let acceptPreviews = verifySources && resultsCurrent
        return DataNodeSchemaProjection.upstreamSchema(for:id,in:project.graph,inputPort:inputPort,sourceSchema:{ sourceID in
            if policy == .normalizationLesson { return policy.fixtureColumns }
            let columns: [String]
            if verifySources { columns = sourceHeaderColumns(sourceID) }
            else if let evidence=sourceHeaders[sourceID], project.sourceBindings[sourceID] == evidence.path,
                    project.graph.nodes.first(where:{$0.id == sourceID})?.config == evidence.config {
                // Recorded guidance only. Rendering does not acquire a grant or reread a file.
                columns = evidence.columns
            } else { columns = [] }
            return columns.map { DataColumnSchema(name:$0,dtype:"string",nullable:nil) }
        },capturedSchema:{ endpoint in
            guard acceptPreviews, let output=self.previews[endpoint.node]?[endpoint.port] as? [String:Any],
                  let preview=output["preview"] as? [String:Any], let columns=preview["columns"] as? [[String:Any]] else {return nil}
            return columns.compactMap { column in
                guard let name=column["name"] as? String, let dtype=column["dtype"] as? String else {return nil}
                return DataColumnSchema(name:name,dtype:dtype,nullable:column["nullable"] as? Bool, temporal:(column["temporal"] as? [String:Any]).flatMap { object in (try? JSONSerialization.data(withJSONObject:object)).flatMap { try? JSONDecoder().decode([String:JSONValue].self,from:$0) } })
            }
        })
    }
}
