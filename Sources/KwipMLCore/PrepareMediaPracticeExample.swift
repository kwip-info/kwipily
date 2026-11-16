import Foundation

/// Public synthetic files only. Source paths are bound privately by the loader.
public enum PrepareMediaPracticeExample {
    public static let exampleID="prepare-media"
    public static let imageSource=WorkflowNodeAddress(domain:.data,localID:"source")
    public static let audioSource=WorkflowNodeAddress(domain:.data,localID:"audio-source")
    public static let imagePrepare=WorkflowNodeAddress(domain:.data,localID:"image-prepare")
    public static let audioPrepare=WorkflowNodeAddress(domain:.data,localID:"audio-prepare")
    public static let resourceNames=["file-practice-12-colors.png","media-practice-tone.wav"]
    public static func document(projectID:String=UUID().uuidString)throws->WorkflowDocument {
        var image=FileInputConfiguration.newDefaults;image["sourceMode"] = .string("directory");image["extensions"] = .array([.string("png")])
        var audio=FileInputConfiguration.newDefaults;audio["sourceMode"] = .string("directory");audio["extensions"] = .array([.string("wav")])
        if case .object(var readers)?=audio["readers"] {readers["audio"] = .object(["format":.string("pcmWav")]);audio["readers"] = .object(readers)}
        var pixels=try PrepareMediaConfiguration.defaults();pixels["sizeMode"] = .string("fitPad");pixels["width"] = 4;pixels["height"] = 4;pixels["sampleFormat"] = .string("float32");pixels["scaling"] = .string("unit");pixels["layout"] = .string("CHW")
        var samples=try PrepareMediaConfiguration.defaults(kind:"audio");samples["sampleFormat"] = .string("float32");samples["windowSamples"] = 256;samples["hopSamples"] = 256;samples["tail"] = .string("pad")
        let imageView=WorkflowNodeAddress(domain:.data,localID:"image-view"),audioView=WorkflowNodeAddress(domain:.data,localID:"audio-view")
        let nodes:[WorkflowNode]=[
            .init(address:imageSource,name:"Read color image",x:50,y:80,payload:.dataOperator(operatorID:"kwip.data.sourceFiles",config:image)),
            .init(address:imagePrepare,name:"Prepare image pixels",x:420,y:80,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:pixels)),
            .init(address:imageView,name:"Inspect prepared image",x:790,y:80,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:audioSource,name:"Read PCM WAV",x:50,y:340,payload:.dataOperator(operatorID:"kwip.data.sourceFiles",config:audio)),
            .init(address:audioPrepare,name:"Prepare audio windows",x:420,y:340,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:samples)),
            .init(address:audioView,name:"Inspect audio and masks",x:790,y:340,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))]
        let pairs=[(imageSource,imagePrepare),(imagePrepare,imageView),(audioSource,audioPrepare),(audioPrepare,audioView)]
        let edges=pairs.enumerated().map {i,pair in WorkflowEdge(id:"media-"+String(i),from:.init(address:pair.0,port:"table"),to:.init(address:pair.1,port:"table"),kind:.table)}
        let document=WorkflowDocument(projectID:projectID,name:"Prepare image pixels and audio windows",nodes:nodes,edges:edges)
        try WorkflowCodec.validate(document);return document
    }
}
