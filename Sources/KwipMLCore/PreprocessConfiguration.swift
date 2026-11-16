import Foundation

public struct PreprocessConfiguration:Equatable,Codable {
    public struct Column:Equatable,Codable,Identifiable {
        public var column:String
        public var missing="keep"
        public var value:JSONValue = 0
        public var emptyPolicy="error"
        public var emptyValue:JSONValue = 0
        public var tiePolicy="error"
        public var encoding="none"
        public var vocabularyOrder="lexicographic"
        public var unknownPolicy="error"
        public var maxCategories=32
        public var scaling="none"
        public var center=true
        public var unitVariance=true
        public var unicodeNormalization="none"
        public var removeControls=false
        public var removePunctuation=false
        public var removeDigits=false
        public var whitespace="none"
        public var casePolicy="preserve"
        public var emptyAfterCleanup="keep"
        public var hasTextCleanup:Bool {unicodeNormalization != "none" || removeControls || removePunctuation || removeDigits || whitespace != "none" || casePolicy != "preserve" || emptyAfterCleanup != "keep"}
        public var id:String {column}
        public init(column:String) {self.column=column}
        enum CodingKeys:String,CodingKey {case column,missing,value,emptyPolicy,emptyValue,tiePolicy,encoding,vocabularyOrder,unknownPolicy,maxCategories,scaling,center,unitVariance,unicodeNormalization,removeControls,removePunctuation,removeDigits,whitespace,emptyAfterCleanup;case casePolicy="case"}
        public func encode(to encoder:Encoder)throws {
            var c=encoder.container(keyedBy:CodingKeys.self)
            try c.encode(column,forKey:.column)
            try c.encode(missing,forKey:.missing)
            try c.encode(value,forKey:.value)
            try c.encode(emptyPolicy,forKey:.emptyPolicy)
            try c.encode(emptyValue,forKey:.emptyValue)
            try c.encode(tiePolicy,forKey:.tiePolicy)
            try c.encode(encoding,forKey:.encoding)
            try c.encode(vocabularyOrder,forKey:.vocabularyOrder)
            try c.encode(unknownPolicy,forKey:.unknownPolicy)
            try c.encode(maxCategories,forKey:.maxCategories)
            try c.encode(scaling,forKey:.scaling)
            try c.encode(center,forKey:.center)
            try c.encode(unitVariance,forKey:.unitVariance)
            if unicodeNormalization != "none" {try c.encode(unicodeNormalization,forKey:.unicodeNormalization)}
            if removeControls != false {try c.encode(removeControls,forKey:.removeControls)}
            if removePunctuation != false {try c.encode(removePunctuation,forKey:.removePunctuation)}
            if removeDigits != false {try c.encode(removeDigits,forKey:.removeDigits)}
            if whitespace != "none" {try c.encode(whitespace,forKey:.whitespace)}
            if casePolicy != "preserve" {try c.encode(casePolicy,forKey:.casePolicy)}
            if emptyAfterCleanup != "keep" {try c.encode(emptyAfterCleanup,forKey:.emptyAfterCleanup)}
        }
        public init(from decoder:Decoder)throws {
            let c=try decoder.container(keyedBy:CodingKeys.self)
            column=try c.decode(String.self,forKey:.column)
            missing=try c.decodeIfPresent(String.self,forKey:.missing) ?? "keep"
            value=try c.decodeIfPresent(JSONValue.self,forKey:.value) ?? 0
            emptyPolicy=try c.decodeIfPresent(String.self,forKey:.emptyPolicy) ?? "error"
            emptyValue=try c.decodeIfPresent(JSONValue.self,forKey:.emptyValue) ?? 0
            tiePolicy=try c.decodeIfPresent(String.self,forKey:.tiePolicy) ?? "error"
            encoding=try c.decodeIfPresent(String.self,forKey:.encoding) ?? "none"
            vocabularyOrder=try c.decodeIfPresent(String.self,forKey:.vocabularyOrder) ?? "lexicographic"
            unknownPolicy=try c.decodeIfPresent(String.self,forKey:.unknownPolicy) ?? "error"
            maxCategories=try c.decodeIfPresent(Int.self,forKey:.maxCategories) ?? 32
            scaling=try c.decodeIfPresent(String.self,forKey:.scaling) ?? "none"
            center=try c.decodeIfPresent(Bool.self,forKey:.center) ?? true
            unitVariance=try c.decodeIfPresent(Bool.self,forKey:.unitVariance) ?? true
            unicodeNormalization=try c.decodeIfPresent(String.self,forKey:.unicodeNormalization) ?? "none"
            removeControls=try c.decodeIfPresent(Bool.self,forKey:.removeControls) ?? false
            removePunctuation=try c.decodeIfPresent(Bool.self,forKey:.removePunctuation) ?? false
            removeDigits=try c.decodeIfPresent(Bool.self,forKey:.removeDigits) ?? false
            whitespace=try c.decodeIfPresent(String.self,forKey:.whitespace) ?? "none"
            casePolicy=try c.decodeIfPresent(String.self,forKey:.casePolicy) ?? "preserve"
            emptyAfterCleanup=try c.decodeIfPresent(String.self,forKey:.emptyAfterCleanup) ?? "keep"
        }

    }
    public var mode:String
    public var columns:[Column]
    public init(mode:String="partitions",columns:[Column]=[]) {self.mode=mode;self.columns=columns}
    public init(values:[String:JSONValue])throws {
        guard Set(values.keys)==["mode","columns"],case .array(let rows)=values["columns"] else {throw DataFlowContractError("Use only mode and column rules.")}
        let keys:Set<String>=["column","missing","value","emptyPolicy","emptyValue","tiePolicy","encoding","vocabularyOrder","unknownPolicy","maxCategories","scaling","center","unitVariance"]
        let cleanup:Set<String>=["unicodeNormalization","removeControls","removePunctuation","removeDigits","whitespace","case","emptyAfterCleanup"]
        for row in rows {guard case .object(let fields)=row,keys.isSubset(of:Set(fields.keys)),Set(fields.keys).isSubset(of:keys.union(cleanup)) else {throw DataFlowContractError("Preprocess column controls are incomplete or unknown.")}}
        self=try JSONDecoder().decode(Self.self,from:JSONEncoder().encode(JSONValue.object(values)))
        try validate()
    }
    public var values:[String:JSONValue] {
        guard case .object(let value)=try! JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(self)) else {return [:]}
        return value
    }
    public func validate()throws {
        func require(_ condition:Bool,_ message:String)throws {if !condition {throw DataFlowContractError(message)}}
        try require(["table","partitions","frozen"].contains(mode) && columns.count<=128 && Set(columns.map(\.column)).count==columns.count,"Choose a mode and unique column rules.")
        try require(mode != "frozen" || columns.isEmpty,"Apply saved rules uses immutable model-bound rules; clear editable column rules.")
        for row in columns {
            for value in [row.value,row.emptyValue] {
                switch value {case .string,.bool,.integer:break;case .number(let n) where n.isFinite:break;default:throw DataFlowContractError("Use finite scalar replacement values.")}
            }
            try require(!row.column.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && row.column.count<=200,"Choose an exact column name.")
            try require(["keep","error","drop","constant","mean","median","mostFrequent"].contains(row.missing) && ["none","oneHot"].contains(row.encoding) && ["none","standard"].contains(row.scaling),"Choose supported missing, encoding and scaling policies.")
            try require(["error","constant"].contains(row.emptyPolicy) && ["error","smallest"].contains(row.tiePolicy) && ["error","ignore"].contains(row.unknownPolicy) && ["lexicographic","firstSeen"].contains(row.vocabularyOrder) && (1...128).contains(row.maxCategories),"Choose valid empty, tie, category and vocabulary policies.")
            try require(mode != "table" || (!["mean","median","mostFrequent"].contains(row.missing) && row.encoding=="none" && row.scaling=="none"),"Learned imputation, encoding and scaling require Training/Validation/Test mode.")
            try require(["none","NFC","NFKC"].contains(row.unicodeNormalization) && ["none","trim","collapse"].contains(row.whitespace) && ["preserve","lower","upper"].contains(row.casePolicy) && ["keep","missing"].contains(row.emptyAfterCleanup),"Choose supported Text cleanup controls.")
            try require(mode=="table" || !row.hasTextCleanup,"Text cleanup requires fixed Table mode before Partition.")
            try require(row.encoding=="none" || row.scaling=="none","Choose encoding or scaling for a source column.")

        }
    }
    public static func matches(_ value:JSONValue,dtype:String)->Bool {
        switch (dtype,value) {case ("string",.string),("bool",.bool),("int64",.integer):return true;case ("float64",.integer):return true;case ("float64",.number(let n)):return n.isFinite;default:return false}
    }
}
