import Foundation
import KwipMLCore
import KwipMLDatabaseNetworking

@objc public protocol KwipMLNetworkProtocol {
    func perform(_ message: Data, reply: @escaping (Data?, String?) -> Void)
    func modelHub(_ message:Data,reply:@escaping(Data?,String?)->Void)
}

public enum NetworkServiceIdentity {
    public static var identifier:String {
        #if DEBUG
        var app=Bundle.main.bundleURL
        while app.pathExtension != "app",app.path != "/" {app.deleteLastPathComponent()}
        if app.pathExtension=="app",let id=Bundle(url:app)?.bundleIdentifier {return id+".NetworkService"}
        #endif
        return DesktopDeployment.networkServiceIdentifier
    }
    public static func clientRequirement()throws->String {
        #if DEBUG
        return try DatabaseSourceServiceIdentity.clientRequirement()
        #else
        return requirement(identifier:DesktopDeployment.bundleIdentifier)
        #endif
    }
    public static func requirement(identifier: String) -> String {
        // Exact code identity prevents same-team worker executables from becoming service peers.
        "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"MT68X9Z6NQ\""
    }
    public static func connect() -> NSXPCConnection {
        let connection = NSXPCConnection(serviceName: identifier)
        connection.remoteObjectInterface = NSXPCInterface(with: KwipMLNetworkProtocol.self)
        #if DEBUG
        var app=Bundle.main.bundleURL
        while app.pathExtension != "app",app.path != "/" {app.deleteLastPathComponent()}
        if app.pathExtension=="app",let exact=try? DatabaseSourceServiceIdentity.designatedRequirement(app.appendingPathComponent("Contents/XPCServices/NetworkService.xpc")) {connection.setCodeSigningRequirement(exact)}
        else {connection.setCodeSigningRequirement(requirement(identifier:identifier))}
        #else
        connection.setCodeSigningRequirement(requirement(identifier: identifier))
        #endif
        connection.resume()
        return connection
    }
}

/// Testable transport with no persistent cache/cookies and a strict response byte bound.
public final class BoundedNetworkTransport: NSObject, URLSessionTaskDelegate {
    #if DEBUG
    var testProtocolClasses: [AnyClass]?
    #endif
    public func urlSession(_ session: URLSession, task: URLSessionTask,
                           willPerformHTTPRedirection response: HTTPURLResponse,
                           newRequest request: URLRequest,
                           completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
    public func perform(_ message: Data) async throws -> Data {
        let request = try NetworkBrokerPolicy.prepare(message)
        let config = URLSessionConfiguration.ephemeral
        #if DEBUG
        if let classes = testProtocolClasses { config.protocolClasses = classes }
        #endif
        config.httpCookieStorage = nil
        config.httpShouldSetCookies = false
        config.urlCache = nil
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 20
        config.timeoutIntervalForResource = 20
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let (bytes, response) = try await session.bytes(for: request)
        if let http=response as? HTTPURLResponse,http.statusCode==401 {throw NetworkBrokerFailure.signInRequired}
        guard let http = response as? HTTPURLResponse, ((200...299).contains(http.statusCode) || [403,409,429].contains(http.statusCode)),
              http.url == request.url,
              response.expectedContentLength <= NetworkBrokerPolicy.maximumResponseBytes else {
            throw NetworkBrokerPolicy.Refusal.invalidMessage
        }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < NetworkBrokerPolicy.maximumResponseBytes else { throw NetworkBrokerPolicy.Refusal.invalidMessage }
            data.append(byte)
        }
        if !(200...299).contains(http.statusCode) {
            if let error=try? JSONSerialization.jsonObject(with:data) as? [String:Any],let code=error["code"] as? String {
                switch (http.statusCode,code) {
                case (403,"not_entitled"):throw NetworkBrokerFailure.notEntitled
                case (409,"device_limit"):throw NetworkBrokerFailure.deviceLimit
                case (409,"conflict"):throw NetworkBrokerFailure.conflict
                case (429,"rate_limited"):throw NetworkBrokerFailure.rateLimited
                default:break
                }
            }
            throw NetworkBrokerFailure.unavailable
        }
        return data
    }
}
