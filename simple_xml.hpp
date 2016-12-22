#ifndef SIMPLE_XML
#define SIMPLE_XML

#include <QDomElement>
#include <QDomDocument>
#include <QDomText>
#include <QDomNodeList>
#include <QDomNode>
#include <QMap>
#include <QVariant>

class SimpleXML{
public:
	QString m_fileName;
	QDomDocument dom;
	QDomNodeList dom_list;
	/// \brief parent node tree with tag
	QDomNode tree_node;
	QString tree_tag;

	/**
	 * @brief SimpleXML
	 * @param filename
	 * @param forSave		if true then constructor attempt load the file "fileName"
	 * @param tag			if file not exists and not loaded and forSave == "true"
	 *						then create being xml with tree tag
	 */
	SimpleXML(const QString& filename, bool forSave = false, const QString& tag = "tree");
	/// \brief load from file
	bool load();
	/// \brief save to file
	void save();
	/// \brief create <?xml version="1.0"?>
	void create_processing_instruction();
	/// \brief create node for '_dom' and return 'node'
	QDomNode create_tree(QDomNode& _dom, QDomNode& node, const QString& tag);
	/**
	 * @brief create_tree
	 * @param tag
	 * @return
	 */
	QDomNode create_tree(const QString& tag = "tree");
	/// \brief create tag with text = string value
	void set_dom_value_s(QDomNode& node, const QString& tag, const QString& value);
	/**
	 * @brief set_dom_value_s
	 * @param tag
	 * @param value
	 */
	void set_dom_value_s(const QString& tag, const QString& value);
	/// \brief create tag with text = unsigned integer value
	void set_dom_value_num(QDomNode& node, const QString& tag, double value);
	/**
	 * @brief set_dom_value_num
	 * @param tag
	 * @param value
	 */
	void set_dom_value_num(const QString& tag, double value);
	/// \brief прочитать строку из xml
	QString get_xml_string(const QString tag, QDomNode* node = NULL);
	/// \brief read unsigned integer value from tag
	uint get_xml_uint(const QString tag, QDomNode* node = NULL);
	/// \brief read integer value from tag
	int get_xml_int(const QString tag, QDomNode* node = NULL);
	/// \brief read double value from tag
	double get_xml_double(const QString tag, QDomNode* node = NULL);
	/// \brief get list attributes
	QDomNamedNodeMap get_list_attributes(const QString tag, QDomNode* node = NULL);
	/// \brief read unsigned integer value from tag
	uint get_xml_uint(QDomNamedNodeMap node, const QString& tag);
	/// \brief read integer value from tag
	int get_xml_int(QDomNamedNodeMap node, const QString& tag);
	/// \brief read double value from tag
	double get_xml_double(QDomNamedNodeMap node, const QString& tag);
	/**
	 * @brief first_child
	 * @return
	 */
	QDomNode first_child();
	/// \brief get node list form xml
	QDomNodeList& get_xml_list(const QString node);
	/// \brief get count nodes
	int count_from_list();
	/// \brief get node form list
	QDomNode get_xml_node_from_list(int index);
	/// \brief read int from node of list
	int get_xml_int(int index);
	/// \brief read string from node of list
	QString get_xml_string(int index);

	/**
	 * @brief get_node
	 * get or create node with tag if it not exists
	 * @param parent_node
	 * @param tag
	 * @return
	 */
	QDomNode get_node(QDomNode& parent_node, const QString& tag);
	/**
	 * @brief save_param
	 * @param fileName
	 * @param params
	 */
	static void save_param(const QString &fileName, const QMap< QString, QVariant>& params);
	/**
	 * @brief load_param
	 * @param fileName
	 * @param params
	 * @return
	 */
	static bool load_param(const QString &fileName, QMap< QString, QVariant>& params);

protected:
	/**
	 * @brief set_tag_value
	 * set value to node with tag. if node not exists then it created
	 * @param tag
	 * @param value
	 * @param parent
	 * @param index
	 */
	void set_tag_value(const QString& tag, const QString& value, QDomNode* parent = NULL, int index = 0);
};

#endif
