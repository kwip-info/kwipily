import Foundation
import KwipMLCore

@objc public protocol MCPAppCommandProtocol {
    func dispatch(_ envelope:Data,reply:@escaping(Data?,String?)->Void)
    func receiveWebhook(_ request:Data,reply:@escaping(Data?,String?)->Void)
}
@objc public protocol MCPServiceProtocol {
    func configure(_ setup:Data,reply:@escaping(Data?,String?)->Void)
    func configureWebhook(_ port:Int,reply:@escaping(Int,String?)->Void)
    func stop(reply:@escaping()->Void)
}
public enum MCPServiceIdentity {
    public static let productionIdentifier="info.kwip.KwipML.MCPService"
    public static let developmentIdentifier="info.kwip.KwipML.POC.MCPService"
    public static var identifier:String {
        #if DEBUG
        return developmentIdentifier
        #else
        return DesktopDeployment.bundleIdentifier+".MCPService"
        #endif
    }
    public static var appIdentifier:String {
        #if DEBUG
        return "info.kwip.KwipML.POC"
        #else
        return DesktopDeployment.bundleIdentifier
        #endif
    }
    public static func requirement(_ identifier:String)->String {
        #if DEBUG
        // Disposable ad-hoc development bundles only. Release never compiles this
        // requirement; production binds exact identifiers and the trusted team.
        return "identifier \"\(identifier)\""
        #else
        return "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"MT68X9Z6NQ\""
        #endif
    }
    public static func connection(appCommands:MCPAppCommandProtocol)->NSXPCConnection {
        let connection=NSXPCConnection(serviceName:identifier)
        connection.setCodeSigningRequirement(requirement(identifier))
        connection.remoteObjectInterface=NSXPCInterface(with:MCPServiceProtocol.self)
        connection.exportedInterface=NSXPCInterface(with:MCPAppCommandProtocol.self)
        connection.exportedObject=appCommands
        connection.resume();return connection
    }
}
