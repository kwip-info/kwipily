import Foundation
import KwipMLCore

/// Explicit authored graph identities; display labels are never used to guess a target.
enum LessonNodeLink {
    static func address(_ url:URL)->WorkflowNodeAddress? {
        guard url.scheme=="kwip-node",let parts=URLComponents(url:url,resolvingAgainstBaseURL:false),
              parts.user==nil,parts.password==nil,parts.port==nil,parts.query==nil,parts.fragment==nil,
              let host=parts.host,let domain=WorkflowDomain(rawValue:host) else {return nil}
        let path=parts.path.split(separator:"/",omittingEmptySubsequences:true)
        guard path.count==1 else {return nil}
        let id=String(path[0])
        guard !id.isEmpty,id.utf8.count<=128,!id.contains("\\"),!id.unicodeScalars.contains(where:{$0.value<32}),id != ".",id != ".." else {return nil}
        return .init(domain:domain,localID:id)
    }
}
